import streamlit as st
import cv2
import tempfile
import subprocess
import os
from pathlib import Path
from collections import Counter
from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Border Surveillance AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM PAGE STYLE
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 22px;
        font-weight: 500;
        margin-bottom: 25px;
    }

    .summary-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
        background-color: #fafafa;
    }

    .summary-number {
        font-size: 34px;
        font-weight: 700;
    }

    .summary-label {
        font-size: 16px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ Smart Border Surveillance AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Human vs Animal Intrusion Detection</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

MODEL_PATH = Path(__file__).parent / "yolov8n.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()
except Exception as e:
    st.error("❌ Unable to load detection model.")
    st.code(str(e))
    st.stop()


# =========================================================
# ANIMAL CLASSES
# =========================================================

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
    "giraffe"
}

CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.30
MAX_MISSED_FRAMES = 12


# =========================================================
# SIMPLE IOU TRACKER
# No model.track(), no ByteTrack, no LAP dependency.
# =========================================================

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def update_tracks(tracks, detections, next_id):
    """
    Match current detections with previous tracks using IoU.

    Each track stores:
        id
        bbox
        category: human / animal
        class_history
        missed
    """

    matched_track_ids = set()
    matched_detection_ids = set()

    # Highest-confidence detections get matched first.
    detections_sorted = sorted(
        enumerate(detections),
        key=lambda item: item[1]["confidence"],
        reverse=True
    )

    # ---------------------------------------------------------
    # Match existing tracks to current detections
    # ---------------------------------------------------------
    for det_index, detection in detections_sorted:

        best_track_id = None
        best_iou = IOU_THRESHOLD

        for track_id, track in tracks.items():

            if track_id in matched_track_ids:
                continue

            if track["category"] != detection["category"]:
                continue

            iou = calculate_iou(
                track["bbox"],
                detection["bbox"]
            )

            if iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is not None:

            track = tracks[best_track_id]

            track["bbox"] = detection["bbox"]
            track["confidence"] = detection["confidence"]
            track["missed"] = 0

            track["class_history"][
                detection["class_name"]
            ] += 1

            matched_track_ids.add(best_track_id)
            matched_detection_ids.add(det_index)

    # ---------------------------------------------------------
    # Create new tracks for unmatched detections
    # ---------------------------------------------------------
    for det_index, detection in enumerate(detections):

        if det_index in matched_detection_ids:
            continue

        track_id = next_id
        next_id += 1

        tracks[track_id] = {
            "id": track_id,
            "bbox": detection["bbox"],
            "category": detection["category"],
            "confidence": detection["confidence"],
            "class_history": Counter(
                {detection["class_name"]: 1}
            ),
            "missed": 0
        }

        matched_track_ids.add(track_id)

    # ---------------------------------------------------------
    # Age tracks that were not detected in this frame
    # ---------------------------------------------------------
    tracks_to_remove = []

    for track_id, track in tracks.items():

        if track_id not in matched_track_ids:
            track["missed"] += 1

            if track["missed"] > MAX_MISSED_FRAMES:
                tracks_to_remove.append(track_id)

    for track_id in tracks_to_remove:
        del tracks[track_id]

    return tracks, next_id


# =========================================================
# MAIN TWO-COLUMN LAYOUT
# =========================================================

left_col, right_col = st.columns(
    [1, 2],
    gap="large"
)


# =========================================================
# LEFT SIDE - SYSTEM CONTROLS
# =========================================================

with left_col:

    st.header("⚙️ System Controls")

    st.markdown("### 🎥 Upload Surveillance Video")

    uploaded_video = st.file_uploader(
        "Choose a surveillance video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    st.markdown("")

    if uploaded_video is not None:

        start_detection = st.button(
            "🚀 Start Detection",
            type="primary",
            use_container_width=True
        )

    else:
        start_detection = False


# =========================================================
# RIGHT SIDE - UPLOADED VIDEO
# =========================================================

with right_col:

    if uploaded_video is not None:

        st.subheader("🎬 Uploaded Surveillance Video")

        st.video(
            uploaded_video,
            width=700
        )


# =========================================================
# DETECTION
# =========================================================

if uploaded_video is not None and start_detection:

    # =====================================================
    # SAVE INPUT VIDEO
    # =====================================================

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        uploaded_video.getbuffer()
    )
    input_file.close()

    input_path = input_file.name


    # =====================================================
    # OPEN VIDEO
    # =====================================================

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():

        st.error("❌ Could not open uploaded video.")

        if os.path.exists(input_path):
            os.remove(input_path)

        st.stop()


    # =====================================================
    # VIDEO INFORMATION
    # =====================================================

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )


    # =====================================================
    # TEMPORARY RAW OUTPUT
    # =====================================================

    raw_output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )

    raw_output.close()

    raw_output_path = raw_output.name


    # =====================================================
    # VIDEO WRITER
    # =====================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"XVID"
    )

    out = cv2.VideoWriter(
        raw_output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():

        cap.release()

        if os.path.exists(input_path):
            os.remove(input_path)

        if os.path.exists(raw_output_path):
            os.remove(raw_output_path)

        st.error("❌ Could not create output video.")
        st.stop()


    # =====================================================
    # TRACKING STATE
    # =====================================================

    tracks = {}
    next_track_id = 1

    # These are final unique IDs seen during the video.
    human_ids = set()
    animal_ids = set()

    # Final species history for unique animal IDs.
    animal_species_by_id = {}


    # =====================================================
    # PROGRESS
    # =====================================================

    progress = st.progress(0)
    status = st.empty()


    # =====================================================
    # FRAME PROCESSING
    # =====================================================

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # -------------------------------------------------
        # YOLO DETECTION ONLY
        # Important: model.track() is NOT used.
        # -------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                class_name = model.names[
                    class_id
                ]

                confidence = float(
                    box.conf[0]
                )

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # -------------------------------------------------
                # HUMAN / ANIMAL CATEGORY
                # -------------------------------------------------

                if class_name == "person":

                    category = "human"

                elif class_name in ANIMAL_CLASSES:

                    category = "animal"

                else:

                    continue

                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "class_name": class_name,
                        "confidence": confidence,
                        "category": category
                    }
                )


        # -------------------------------------------------
        # UPDATE UNIQUE OBJECT TRACKS
        # -------------------------------------------------

        tracks, next_track_id = update_tracks(
            tracks,
            detections,
            next_track_id
        )


        # -------------------------------------------------
        # DRAW CURRENT TRACKS / BOUNDING BOXES
        # -------------------------------------------------

        for track_id, track in tracks.items():

            if track["missed"] > 0:
                continue

            x1, y1, x2, y2 = track["bbox"]

            category = track["category"]
            confidence = track["confidence"]

            # Most frequently predicted class for this track.
            class_name = track["class_history"].most_common(1)[0][0]

            if category == "human":

                human_ids.add(track_id)

                label = (
                    f"HUMAN #{track_id} "
                    f"{confidence:.2f}"
                )

                box_color = (0, 255, 0)

            else:

                animal_ids.add(track_id)

                animal_species_by_id[
                    track_id
                ] = track["class_history"]

                label = (
                    f"ANIMAL #{track_id} | "
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

                box_color = (0, 165, 255)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )

            # Label background
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )

            label_y = max(y1 - 8, text_h + 8)

            cv2.rectangle(
                frame,
                (
                    x1,
                    label_y - text_h - baseline - 6
                ),
                (
                    x1 + text_w + 6,
                    label_y + 3
                ),
                box_color,
                -1
            )

            cv2.putText(
                frame,
                label,
                (
                    x1 + 3,
                    label_y - 3
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )


        # -------------------------------------------------
        # WRITE FRAME
        # -------------------------------------------------

        out.write(frame)


        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        if total_frames > 0:

            progress_value = (
                frame_number / total_frames
            )

            progress.progress(
                min(progress_value, 1.0)
            )

            status.write(
                f"Processing frame "
                f"{frame_number}/{total_frames}"
            )


    # =====================================================
    # RELEASE VIDEO
    # =====================================================

    cap.release()
    out.release()

    progress.progress(1.0)

    status.success(
        "✅ Video processing completed!"
    )


    # =====================================================
    # FINAL UNIQUE COUNTS
    # =====================================================

    human_count = len(human_ids)
    animal_count = len(animal_ids)


    # =====================================================
    # FINAL ANIMAL SPECIES BREAKDOWN
    # Count each UNIQUE animal only once.
    # =====================================================

    final_species_counts = Counter()

    for animal_id in animal_ids:

        history = animal_species_by_id.get(
            animal_id,
            Counter()
        )

        if history:

            final_species = history.most_common(1)[0][0]

            final_species_counts[
                final_species
            ] += 1


    # =====================================================
    # CONVERT TO H264 MP4
    # =====================================================

    status.info(
        "🎞️ Preparing processed video..."
    )

    final_output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    final_output.close()

    final_output_path = final_output.name

    try:

        import imageio_ffmpeg

        ffmpeg_path = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            raw_output_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            final_output_path
        ]

        subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )

    except Exception as e:

        st.error(
            "❌ Could not prepare processed video."
        )

        st.code(str(e))

        cap.release()
        out.release()

        st.stop()


    # =====================================================
    # READ FINAL VIDEO INTO MEMORY
    # =====================================================

    with open(
        final_output_path,
        "rb"
    ) as video_file:

        processed_video = video_file.read()


    # =====================================================
    # LEFT SIDE - DETECTION SUMMARY
    # =====================================================

    with left_col:

        st.markdown("---")

        st.header("📊 Detection Summary")

        # -------------------------------------------------
        # HUMAN
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">
                    👤 Humans Detected
                </div>
                <div class="summary-number">
                    {human_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # ANIMAL
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">
                    🐾 Animals Detected
                </div>
                <div class="summary-number">
                    {animal_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # SPECIES BREAKDOWN
        # -------------------------------------------------

        if final_species_counts:

            st.markdown(
                "### 🐾 Animal Breakdown"
            )

            for species, count in sorted(
                final_species_counts.items()
            ):

                st.write(
                    f"• {species.title()}: {count}"
                )


        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        st.download_button(
            label="⬇️ Download Processed Video",
            data=processed_video,
            file_name=(
                "smart_border_surveillance_result.mp4"
            ),
            mime="video/mp4",
            use_container_width=True
        )

        st.success(
            "🛡️ Surveillance analysis completed."
        )


    # =====================================================
    # RIGHT SIDE - PROCESSED VIDEO
    # =====================================================

    with right_col:

        st.markdown("---")

        st.subheader(
            "🎥 Processed Surveillance Video"
        )

        st.video(
            processed_video,
            width=700
        )

        st.success(
            "✅ Bounding boxes and unique-object counting applied."
        )


    # =====================================================
    # CLEAN TEMPORARY FILES
    # =====================================================

    for temporary_file in [
        input_path,
        raw_output_path,
        final_output_path
    ]:

        try:

            if os.path.exists(temporary_file):
                os.remove(temporary_file)

        except Exception:
            pass
