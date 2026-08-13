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

st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🛡️ Smart Border Surveillance AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Human vs Animal Intrusion Detection</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL
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
# SETTINGS
# =========================================================

ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

CONFIDENCE_THRESHOLD = 0.50

# Lower NMS IoU = more aggressive duplicate suppression.
NMS_IOU = 0.40

# Tracking parameters.
TRACK_IOU = 0.15
MAX_MISSED_FRAMES = 18

# Minimum normalized centre-distance similarity used as a
# fallback when IoU becomes small because an object moved.
MAX_CENTER_DISTANCE = 0.18


# =========================================================
# HELPERS
# =========================================================

def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def normalized_center_distance(box_a, box_b, frame_width, frame_height):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    dx = (ax - bx) / max(frame_width, 1)
    dy = (ay - by) / max(frame_height, 1)

    return (dx * dx + dy * dy) ** 0.5


def remove_duplicate_detections(detections, frame_width, frame_height):
    """
    Extra safety layer after YOLO NMS.

    Keeps the highest-confidence detection when two detections
    of the same class/category overlap heavily.
    """
    kept = []

    detections = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    for detection in detections:
        duplicate = False

        for existing in kept:
            if detection["class_name"] != existing["class_name"]:
                continue

            iou = calculate_iou(
                detection["bbox"],
                existing["bbox"]
            )

            center_distance = normalized_center_distance(
                detection["bbox"],
                existing["bbox"],
                frame_width,
                frame_height
            )

            # Same class + strong overlap = same detection.
            if iou >= 0.35:
                duplicate = True
                break

            # Very close boxes with similar object location.
            if center_distance <= 0.035:
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


def update_tracks(tracks, detections, next_id, frame_width, frame_height):
    """
    Lightweight tracker without model.track(), ByteTrack or LAP.

    Matching priority:
    1. Same category/class + IoU
    2. Same category/class + centre distance

    This prevents one moving object from receiving a new ID every
    few frames.
    """

    unmatched_detections = set(range(len(detections)))
    matched_tracks = set()

    candidates = []

    for track_id, track in tracks.items():
        for det_index, detection in enumerate(detections):
            if track["category"] != detection["category"]:
                continue

            # Prefer same species for animals.
            if (
                track["category"] == "animal"
                and track["class_name"] != detection["class_name"]
            ):
                continue

            iou = calculate_iou(
                track["bbox"],
                detection["bbox"]
            )

            distance = normalized_center_distance(
                track["bbox"],
                detection["bbox"],
                frame_width,
                frame_height
            )

            # Combined matching score.
            if iou >= TRACK_IOU or distance <= MAX_CENTER_DISTANCE:
                score = iou + (1.0 - min(distance / MAX_CENTER_DISTANCE, 1.0)) * 0.35
                candidates.append(
                    (score, track_id, det_index)
                )

    candidates.sort(reverse=True)

    for score, track_id, det_index in candidates:
        if track_id in matched_tracks:
            continue
        if det_index not in unmatched_detections:
            continue

        detection = detections[det_index]
        track = tracks[track_id]

        track["bbox"] = detection["bbox"]
        track["confidence"] = detection["confidence"]
        track["class_name"] = detection["class_name"]
        track["class_history"][detection["class_name"]] += 1
        track["missed"] = 0

        matched_tracks.add(track_id)
        unmatched_detections.remove(det_index)

    # Create new tracks.
    for det_index in unmatched_detections:
        detection = detections[det_index]

        track_id = next_id
        next_id += 1

        tracks[track_id] = {
            "id": track_id,
            "bbox": detection["bbox"],
            "category": detection["category"],
            "class_name": detection["class_name"],
            "confidence": detection["confidence"],
            "class_history": Counter(
                {detection["class_name"]: 1}
            ),
            "missed": 0
        }

    # Age unmatched tracks.
    for track_id in list(tracks.keys()):
        if track_id not in matched_tracks and track_id not in [
            t["id"] for t in tracks.values()
            if t["missed"] == 0
        ]:
            tracks[track_id]["missed"] += 1

    # Remove tracks that disappeared for too long.
    for track_id in list(tracks.keys()):
        if tracks[track_id]["missed"] > MAX_MISSED_FRAMES:
            del tracks[track_id]

    return tracks, next_id


# =========================================================
# LAYOUT
# =========================================================

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.header("⚙️ System Controls")
    st.markdown("### 🎥 Upload Surveillance Video")

    uploaded_video = st.file_uploader(
        "Choose a surveillance video",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed"
    )

    if uploaded_video is not None:
        start_detection = st.button(
            "🚀 Start Detection",
            type="primary",
            use_container_width=True
        )
    else:
        start_detection = False


with right_col:
    if uploaded_video is not None:
        st.subheader("🎬 Uploaded Surveillance Video")
        st.video(uploaded_video, width=700)


# =========================================================
# DETECTION
# =========================================================

if uploaded_video is not None and start_detection:

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )
    input_file.write(uploaded_video.getbuffer())
    input_file.close()
    input_path = input_file.name

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        st.error("❌ Could not open uploaded video.")
        st.stop()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    raw_output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )
    raw_output.close()
    raw_output_path = raw_output.name

    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    out = cv2.VideoWriter(
        raw_output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        st.error("❌ Could not create output video.")
        st.stop()

    # Tracking state.
    tracks = {}
    next_track_id = 1

    # IDs that appeared during the complete video.
    human_ids = set()
    animal_ids = set()

    # Species assigned to each unique animal.
    animal_species_by_id = {}

    progress = st.progress(0)
    status = st.empty()

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # -----------------------------------------------------
        # YOLO DETECTION
        # Explicit NMS IoU is important here.
        # -----------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU,
            max_det=30,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                if class_name == "person":
                    category = "human"
                elif class_name in ANIMAL_CLASSES:
                    category = "animal"
                else:
                    continue

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "class_name": class_name,
                    "confidence": confidence,
                    "category": category
                })

        # Extra duplicate protection.
        detections = remove_duplicate_detections(
            detections,
            width,
            height
        )

        # -----------------------------------------------------
        # TRACK UNIQUE OBJECTS
        # -----------------------------------------------------

        tracks, next_track_id = update_tracks(
            tracks,
            detections,
            next_track_id,
            width,
            height
        )

        # -----------------------------------------------------
        # DRAW BOUNDING BOXES
        # -----------------------------------------------------

        for track_id, track in tracks.items():

            if track["missed"] > 0:
                continue

            x1, y1, x2, y2 = track["bbox"]
            confidence = track["confidence"]
            category = track["category"]

            class_name = (
                track["class_history"]
                .most_common(1)[0][0]
            )

            if category == "human":

                human_ids.add(track_id)

                label = (
                    f"HUMAN #{track_id} | "
                    f"{confidence:.2f}"
                )

                box_color = (0, 255, 0)

            else:

                animal_ids.add(track_id)

                animal_species_by_id[track_id] = (
                    track["class_history"].copy()
                )

                label = (
                    f"ANIMAL #{track_id} | "
                    f"{class_name} | "
                    f"{confidence:.2f}"
                )

                box_color = (0, 165, 255)

            # Bounding box.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )

            # Label.
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2
            )

            label_y = max(
                y1 - 8,
                text_h + 8
            )

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
                (x1 + 3, label_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

        out.write(frame)

        if total_frames > 0:
            progress.progress(
                min(frame_number / total_frames, 1.0)
            )

        status.write(
            f"Processing frame {frame_number}/{total_frames}"
        )

    cap.release()
    out.release()

    progress.progress(1.0)
    status.success("✅ Video processing completed!")


    # =====================================================
    # FINAL UNIQUE COUNTS
    # =====================================================

    human_count = len(human_ids)
    animal_count = len(animal_ids)

    final_species_counts = Counter()

    for animal_id in animal_ids:

        history = animal_species_by_id.get(
            animal_id,
            Counter()
        )

        if history:
            final_species = history.most_common(1)[0][0]
            final_species_counts[final_species] += 1


    # =====================================================
    # CONVERT AVI -> H264 MP4
    # =====================================================

    status.info("🎞️ Preparing processed video...")

    final_output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )
    final_output.close()
    final_output_path = final_output.name

    try:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

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
        st.error("❌ Could not prepare processed video.")
        st.code(str(e))
        st.stop()


    with open(final_output_path, "rb") as video_file:
        processed_video = video_file.read()


    # =====================================================
    # LEFT: SUMMARY
    # =====================================================

    with left_col:

        st.markdown("---")
        st.header("📊 Detection Summary")

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">👤 Humans Detected</div>
                <div class="summary-number">{human_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">🐾 Animals Detected</div>
                <div class="summary-number">{animal_count}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if final_species_counts:

            st.markdown("### 🐾 Animal Breakdown")

            for species, count in sorted(
                final_species_counts.items()
            ):
                st.write(
                    f"• {species.title()}: {count}"
                )

        st.download_button(
            label="⬇️ Download Processed Video",
            data=processed_video,
            file_name="smart_border_surveillance_result.mp4",
            mime="video/mp4",
            use_container_width=True
        )


    # =====================================================
    # RIGHT: PROCESSED VIDEO
    # =====================================================

    with right_col:

        st.markdown("---")
        st.subheader("🎥 Processed Surveillance Video")

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
