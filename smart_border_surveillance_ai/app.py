import streamlit as st
import cv2
import tempfile
import subprocess
import os
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import imageio_ffmpeg


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Border Surveillance AI",
    page_icon="🛡️",
    layout="wide"
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
except Exception as e:
    st.error("❌ YOLO model could not be loaded.")
    st.stop()


# ============================================================
# CLASSES
# ============================================================

ANIMAL_CLASSES = {
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "bird"
}


# ============================================================
# VIDEO UPLOAD
# ============================================================

uploaded_video = st.file_uploader(
    "🎥 Upload Surveillance Video",
    type=["mp4", "avi", "mov", "mkv"]
)


# ============================================================
# MAIN
# ============================================================

if uploaded_video:

    # --------------------------------------------------------
    # Temporary input video
    # --------------------------------------------------------

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(uploaded_video.getbuffer())
    input_file.close()

    input_path = input_file.name

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    left_col, right_col = st.columns(
        [0.32, 0.68],
        gap="large"
    )

    # ========================================================
    # LEFT SIDE
    # ========================================================

    with left_col:

        st.header("⚙️ System Controls")

        st.markdown("### 🎥 Uploaded Video")

        st.video(input_path)

        start_detection = st.button(
            "🚀 Start Detection",
            use_container_width=True
        )

    # ========================================================
    # RIGHT SIDE
    # ========================================================

    with right_col:

        st.header("🎬 Uploaded Surveillance Video")

        st.video(input_path)


    # ========================================================
    # DETECTION
    # ========================================================

    if start_detection:

        with st.spinner("🔍 Analyzing surveillance video..."):

            # ------------------------------------------------
            # Open video
            # ------------------------------------------------

            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():

                st.error("❌ Could not open uploaded video.")
                st.stop()


            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:
                fps = 25


            total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )


            # ------------------------------------------------
            # Temporary processing video
            # ------------------------------------------------

            raw_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            raw_output.close()

            raw_output_path = raw_output.name


            # MP4V temporary writer
            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (width, height)
            )


            # =================================================
            # TRACKING DATA
            # =================================================

            # Track ID -> information
            tracks = {}

            # Confirmed object IDs
            confirmed_humans = set()
            confirmed_animals = set()

            # Animal class -> confirmed IDs
            animal_tracks = defaultdict(set)

            # Track ID -> number of frames seen
            track_frames = defaultdict(int)

            # Track ID -> class
            track_classes = {}

            # Track ID -> last frame
            track_last_seen = {}

            # ------------------------------------------------
            # Confirmation threshold
            # ------------------------------------------------

            # Object must be visible for several frames
            # before being considered a real object.
            MIN_CONFIRM_FRAMES = 8


            # =================================================
            # PROGRESS
            # =================================================

            progress = st.progress(0)

            status = st.empty()


            frame_number = 0


            # =================================================
            # FRAME LOOP
            # =================================================

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1


                # ------------------------------------------------
                # YOLO + BYTE TRACK
                # ------------------------------------------------

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",

                    # Higher confidence reduces false positives
                    conf=0.50,

                    # Stricter NMS
                    iou=0.40,

                    # Only person + animals
                    classes=[
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
                        23   # giraffe
                    ],

                    verbose=False
                )


                result = results[0]


                # =================================================
                # DETECTIONS
                # =================================================

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    boxes = result.boxes


                    for i in range(len(boxes)):

                        # ------------------------------------------------
                        # Track ID
                        # ------------------------------------------------

                        if boxes.id is None:
                            continue

                        track_id = int(
                            boxes.id[i].item()
                        )


                        # ------------------------------------------------
                        # Class
                        # ------------------------------------------------

                        cls_id = int(
                            boxes.cls[i].item()
                        )

                        class_name = model.names[
                            cls_id
                        ]


                        # ------------------------------------------------
                        # Confidence
                        # ------------------------------------------------

                        confidence = float(
                            boxes.conf[i].item()
                        )


                        if confidence < 0.50:
                            continue


                        # ------------------------------------------------
                        # Bounding Box
                        # ------------------------------------------------

                        x1, y1, x2, y2 = map(
                            int,
                            boxes.xyxy[i].tolist()
                        )


                        # =================================================
                        # TRACK CONFIRMATION
                        # =================================================

                        track_frames[track_id] += 1

                        track_classes[
                            track_id
                        ] = class_name

                        track_last_seen[
                            track_id
                        ] = frame_number


                        # ------------------------------------------------
                        # HUMAN
                        # ------------------------------------------------

                        if class_name == "person":

                            if (
                                track_frames[track_id]
                                >= MIN_CONFIRM_FRAMES
                            ):

                                confirmed_humans.add(
                                    track_id
                                )


                            label = (
                                f"HUMAN #{track_id}"
                                f" | {confidence:.2f}"
                            )

                            box_color = (
                                0,
                                255,
                                0
                            )


                        # ------------------------------------------------
                        # ANIMAL
                        # ------------------------------------------------

                        elif class_name in ANIMAL_CLASSES:

                            if (
                                track_frames[track_id]
                                >= MIN_CONFIRM_FRAMES
                            ):

                                confirmed_animals.add(
                                    track_id
                                )

                                animal_tracks[
                                    class_name
                                ].add(track_id)


                            label = (
                                f"ANIMAL #{track_id}"
                                f" | {class_name}"
                                f" {confidence:.2f}"
                            )

                            box_color = (
                                0,
                                165,
                                255
                            )

                        else:
                            continue


                        # =================================================
                        # DRAW BOUNDING BOX
                        # =================================================

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            box_color,
                            3
                        )


                        # ------------------------------------------------
                        # Label background
                        # ------------------------------------------------

                        (text_w, text_h), _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            2
                        )


                        label_y = max(
                            y1 - 8,
                            text_h + 5
                        )


                        cv2.rectangle(
                            frame,
                            (
                                x1,
                                label_y - text_h - 8
                            ),
                            (
                                x1 + text_w + 8,
                                label_y
                            ),
                            box_color,
                            -1
                        )


                        cv2.putText(
                            frame,
                            label,
                            (
                                x1 + 4,
                                label_y - 4
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA
                        )


                # =================================================
                # WRITE FRAME
                # =================================================

                out.write(frame)


                # =================================================
                # PROGRESS
                # =================================================

                if total_frames > 0:

                    value = (
                        frame_number /
                        total_frames
                    )

                    progress.progress(
                        min(value, 1.0)
                    )


                status.write(
                    f"Processing frame "
                    f"{frame_number}/{total_frames}"
                )


            # =================================================
            # RELEASE
            # =================================================

            cap.release()
            out.release()


            progress.progress(1.0)

            status.success(
                "✅ Video processing completed!"
            )


            # =================================================
            # FINAL COUNTS
            # =================================================

            human_count = len(
                confirmed_humans
            )

            animal_count = len(
                confirmed_animals
            )


            # =================================================
            # ANIMAL BREAKDOWN
            # =================================================

            animal_breakdown = {}

            for animal_class, ids in animal_tracks.items():

                valid_ids = [
                    track_id
                    for track_id in ids
                    if track_frames[track_id]
                    >= MIN_CONFIRM_FRAMES
                ]

                if valid_ids:

                    animal_breakdown[
                        animal_class
                    ] = len(
                        set(valid_ids)
                    )


            # =================================================
            # FFMPEG CONVERSION
            # =================================================

            st.info(
                "🎞️ Preparing processed video for browser playback..."
            )


            try:

                ffmpeg_path = (
                    imageio_ffmpeg
                    .get_ffmpeg_exe()
                )


                browser_output = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                browser_output.close()

                browser_output_path = (
                    browser_output.name
                )


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

                    browser_output_path
                ]


                subprocess.run(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True
                )


                processed_video_path = (
                    browser_output_path
                )


            except Exception as e:

                st.warning(
                    "⚠️ Browser conversion failed. "
                    "Using processed video directly."
                )

                processed_video_path = (
                    raw_output_path
                )


        # ========================================================
        # RESULTS
        # ========================================================

        st.divider()

        result_left, result_right = st.columns(
            [0.32, 0.68],
            gap="large"
        )


        # ========================================================
        # LEFT - SUMMARY
        # ========================================================

        with result_left:

            st.header("📊 Detection Summary")


            st.metric(
                "👤 Humans Detected",
                human_count
            )


            st.metric(
                "🐾 Animals Detected",
                animal_count
            )


            # ------------------------------------------------
            # Animal Breakdown
            # ------------------------------------------------

            if animal_breakdown:

                st.subheader(
                    "🐾 Animal Breakdown"
                )

                for animal, count in sorted(
                    animal_breakdown.items()
                ):

                    st.write(
                        f"• **{animal.title()}**: {count}"
                    )

            else:

                st.write(
                    "No animals detected."
                )


            st.success(
                "🛡️ Surveillance analysis completed."
            )


        # ========================================================
        # RIGHT - PROCESSED VIDEO
        # ========================================================

        with result_right:

            st.header(
                "🎥 Processed Surveillance Video"
            )


            try:

                with open(
                    processed_video_path,
                    "rb"
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )


                st.video(
                    video_bytes
                )


                st.download_button(
                    "⬇️ Download Processed Video",
                    data=video_bytes,
                    file_name="processed_surveillance.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )


            except Exception:

                st.error(
                    "❌ Could not display processed video."
                )
