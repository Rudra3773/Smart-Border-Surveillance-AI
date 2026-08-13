import streamlit as st
import cv2
import tempfile
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO
import imageio_ffmpeg


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Border Surveillance AI",
    page_icon="🛡️",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Smart Border Surveillance AI")
st.subheader("Human vs Animal Intrusion Detection")


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "yolov8n.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as exc:
    st.error("❌ YOLO model could not be loaded.")
    st.exception(exc)
    st.stop()


# ============================================================
# CLASSES
# ============================================================

ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}

# COCO class IDs used by YOLOv8
DETECTION_CLASS_IDS = [
    0,   # person
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
]


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Upload Surveillance Video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_video is None:
    st.info("Upload a surveillance video to start the analysis.")
    st.stop()


# Save the uploaded video for this run
input_file = tempfile.NamedTemporaryFile(
    delete=False,
    suffix=".mp4",
)
input_file.write(uploaded_video.getbuffer())
input_file.close()
input_path = input_file.name


# ============================================================
# TOP LAYOUT
# LEFT  = SYSTEM CONTROLS
# RIGHT = UPLOADED VIDEO
# ============================================================

control_col, upload_col = st.columns(
    [0.32, 0.68],
    gap="large",
)

with control_col:

    st.header("⚙️ System Controls")
    st.caption(f"📁 {uploaded_video.name}")

    start_detection = st.button(
        "🚀 Start Detection",
        use_container_width=True,
        type="primary",
    )

with upload_col:

    st.header("🎬 Uploaded Surveillance Video")
    st.video(input_path)


# ============================================================
# DETECTION
# ============================================================

if start_detection:

    with st.spinner("🔍 Analyzing surveillance video..."):

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --------------------------------------------------------
        # Temporary output video
        # --------------------------------------------------------

        raw_output = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        )
        raw_output.close()
        raw_output_path = raw_output.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            raw_output_path,
            fourcc,
            fps,
            (width, height),
        )

        if not out.isOpened():
            cap.release()
            st.error("❌ Could not create the processed video.")
            st.stop()

        # --------------------------------------------------------
        # TRACKING STATE
        #
        # One ByteTrack ID = one tracked object.
        #
        # We keep class history for every ID. This prevents one
        # wrong frame from changing a cow into a horse/elephant.
        # --------------------------------------------------------

        track_frames = defaultdict(int)
        track_class_votes = defaultdict(Counter)
        confirmed_ids = set()

        # Object must remain visible for this many accepted
        # detection frames before it enters the final count.
        MIN_CONFIRM_FRAMES = 12

        # Confidence thresholds.
        # Animal threshold is intentionally higher to reduce
        # false animal detections in surveillance footage.
        PERSON_CONF = 0.55
        ANIMAL_CONF = 0.70

        progress = st.progress(0)
        status = st.empty()

        frame_number = 0

        # ========================================================
        # FRAME LOOP
        # ========================================================

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # ----------------------------------------------------
            # YOLO + BYTE TRACK
            # ----------------------------------------------------

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.50,
                iou=0.30,
                agnostic_nms=True,
                imgsz=960,
                classes=DETECTION_CLASS_IDS,
                verbose=False,
            )

            result = results[0]

            # ----------------------------------------------------
            # DETECTIONS
            # ----------------------------------------------------

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes

                for i in range(len(boxes)):

                    # Unique counting requires a tracker ID.
                    if boxes.id is None:
                        continue

                    track_id = int(boxes.id[i].item())

                    cls_id = int(boxes.cls[i].item())
                    class_name = model.names[cls_id]

                    confidence = float(boxes.conf[i].item())

                    # Class-specific confidence filtering
                    if class_name == "person":

                        if confidence < PERSON_CONF:
                            continue

                    elif class_name in ANIMAL_CLASSES:

                        if confidence < ANIMAL_CONF:
                            continue

                    else:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        boxes.xyxy[i].tolist(),
                    )

                    # ------------------------------------------------
                    # UPDATE TRACK HISTORY
                    # ------------------------------------------------

                    track_frames[track_id] += 1
                    track_class_votes[track_id][class_name] += 1

                    if track_frames[track_id] >= MIN_CONFIRM_FRAMES:
                        confirmed_ids.add(track_id)

                    # Majority-vote class for this object
                    stable_class = (
                        track_class_votes[track_id]
                        .most_common(1)[0][0]
                    )

                    # ------------------------------------------------
                    # DRAW ONLY CONFIRMED OBJECTS
                    # ------------------------------------------------

                    if track_id not in confirmed_ids:
                        continue

                    if stable_class == "person":

                        box_color = (0, 255, 0)

                        label = (
                            f"HUMAN #{track_id} | "
                            f"{confidence:.2f}"
                        )

                    else:

                        box_color = (0, 165, 255)

                        label = (
                            f"ANIMAL #{track_id} | "
                            f"{stable_class} | "
                            f"{confidence:.2f}"
                        )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        3,
                    )

                    # Label size
                    (text_w, text_h), _ = cv2.getTextSize(
                        label,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        2,
                    )

                    label_y = max(
                        y1 - 8,
                        text_h + 8,
                    )

                    # Label background
                    cv2.rectangle(
                        frame,
                        (
                            x1,
                            label_y - text_h - 8,
                        ),
                        (
                            x1 + text_w + 8,
                            label_y,
                        ),
                        box_color,
                        -1,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1 + 4,
                            label_y - 4,
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )

            # ----------------------------------------------------
            # WRITE FRAME
            # ----------------------------------------------------

            out.write(frame)

            # ----------------------------------------------------
            # PROGRESS
            # ----------------------------------------------------

            if total_frames > 0:

                progress.progress(
                    min(
                        frame_number / total_frames,
                        1.0,
                    )
                )

            status.write(
                f"Processing frame "
                f"{frame_number}/{total_frames}"
            )

        # ========================================================
        # RELEASE VIDEO
        # ========================================================

        cap.release()
        out.release()

        progress.progress(1.0)

        status.success(
            "✅ Video processing completed!"
        )

        # ========================================================
        # FINAL UNIQUE COUNTS
        # ========================================================

        human_ids = set()
        animal_ids = set()

        for track_id in confirmed_ids:

            if not track_class_votes[track_id]:
                continue

            stable_class = (
                track_class_votes[track_id]
                .most_common(1)[0][0]
            )

            if stable_class == "person":

                human_ids.add(track_id)

            elif stable_class in ANIMAL_CLASSES:

                animal_ids.add(track_id)

        human_count = len(human_ids)
        animal_count = len(animal_ids)

        # --------------------------------------------------------
        # FINAL ANIMAL BREAKDOWN
        # --------------------------------------------------------

        animal_breakdown = Counter()

        for track_id in animal_ids:

            stable_class = (
                track_class_votes[track_id]
                .most_common(1)[0][0]
            )

            animal_breakdown[stable_class] += 1

        # ========================================================
        # FFMPEG / BROWSER-FRIENDLY MP4
        # ========================================================

        st.info(
            "🎞️ Preparing processed video for browser playback..."
        )

        try:

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

            browser_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            )
            browser_output.close()

            browser_output_path = browser_output.name

            command = [
                ffmpeg_path,
                "-y",
                "-i",
                raw_output_path,
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                browser_output_path,
            ]

            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            processed_video_path = browser_output_path

        except Exception:

            processed_video_path = raw_output_path

            st.warning(
                "⚠️ Browser conversion failed. "
                "Using the processed MP4 directly."
            )

    # ============================================================
    # RESULTS
    # LEFT  = DETECTION SUMMARY
    # RIGHT = PROCESSED VIDEO
    # ============================================================

    st.divider()

    summary_col, video_col = st.columns(
        [0.32, 0.68],
        gap="large",
    )

    # ============================================================
    # LEFT - DETECTION SUMMARY
    # ============================================================

    with summary_col:

        st.header("📊 Detection Summary")

        st.metric(
            "👤 Humans Detected",
            human_count,
        )

        st.metric(
            "🐾 Animals Detected",
            animal_count,
        )

        st.subheader("🐾 Animal Breakdown")

        if animal_breakdown:

            for animal, count in sorted(
                animal_breakdown.items()
            ):

                st.write(
                    f"• **{animal.title()}**: {count}"
                )

        else:

            st.write("No animals detected.")

        st.success(
            "🛡️ Surveillance analysis completed."
        )

    # ============================================================
    # RIGHT - PROCESSED VIDEO
    # ============================================================

    with video_col:

        st.header("🎥 Processed Surveillance Video")

        try:

            with open(
                processed_video_path,
                "rb",
            ) as video_file:

                video_bytes = video_file.read()

            st.video(video_bytes)

            st.download_button(
                "⬇️ Download Processed Video",
                data=video_bytes,
                file_name="processed_surveillance.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                "❌ Could not display processed video."
            )
            st.exception(exc)
