"""Streamlit Web Dashboard for Katomaran Intelligent Face Tracker.

Provides:
  - Upload Media section: still-image face detection/recognition and full
    video pipeline processing (YOLO → ArcFace → Tracker → Events → DB).
  - Real-time KPI metrics (unique visitors, entries, exits, in-scene estimate).
  - Registered Visitors table.
  - Event Audit Trail.
  - Face Image Gallery.
  - System Events Log.

Run with:
    .\\venv\\Scripts\\streamlit.exe run dashboard.py

The dashboard reads from the live SQLite database.  Starting the dashboard
does NOT start the video processing pipeline; the two are independent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.database import DatabaseManager

st.set_page_config(
    page_title="Katomaran Face Tracker Dashboard",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resource loaders
# Models are expensive to initialise. Cache them across Streamlit reruns so
# that image uploads don't re-load weights on every button click.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading YOLO face detector…")
def _get_detector(yolo_path: str, conf: float, min_size: int):
    """Load and cache the YOLO FaceDetector for the lifetime of the session."""
    try:
        from src.face_detector import FaceDetector
        return FaceDetector(
            model_path=yolo_path,
            confidence_threshold=conf,
            min_face_size=min_size,
        )
    except Exception as exc:
        st.warning(f"Face detector unavailable: {exc}")
        return None


@st.cache_resource(show_spinner="Loading InsightFace recognizer…")
def _get_recognizer(model_name: str, threshold: float, min_size: int):
    """Load and cache the ArcFace FaceRecognizer for the lifetime of the session."""
    try:
        from src.face_recognizer import FaceRecognizer
        return FaceRecognizer(
            model_name=model_name,
            similarity_threshold=threshold,
            min_face_size=min_size,
        )
    except Exception as exc:
        st.warning(f"Face recognizer unavailable: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_app_data():
    """Load application config and open the database manager."""
    config = load_config()
    db_path = config.get_resolved_database_path()
    db = DatabaseManager(db_path) if db_path.exists() else None
    return config, db


# ---------------------------------------------------------------------------
# Result renderers
# ---------------------------------------------------------------------------

def _render_image_result(result: Dict[str, Any]) -> None:
    """Display face detection / recognition results for an uploaded still image."""
    if not result.get("success"):
        st.error(f"Image processing failed: {result.get('error', 'Unknown error')}")
        return

    face_count: int = result.get("face_count", 0)
    faces: List[Dict[str, Any]] = result.get("faces", [])
    annotated = result.get("annotated_image")

    col_img, col_info = st.columns([3, 2])

    with col_img:
        if annotated is not None:
            st.image(
                annotated,
                caption="Detected faces  (🟢 green = recognized · 🟠 orange = unknown / embedding failed)",
                use_container_width=True,
            )

    with col_info:
        st.metric("Faces Detected", face_count)

        if face_count == 0:
            st.info("No faces detected. Try a clearer image or lower the confidence threshold in `config.json`.")
        else:
            for i, face in enumerate(faces, 1):
                fid = face.get("face_id")
                is_new = face.get("is_new", False)
                sim = face.get("similarity", 0.0)
                conf = face.get("confidence", 0.0)

                if fid:
                    badge = "🆕 **New visitor registered**" if is_new else "🔄 **Returning visitor**"
                    st.success(
                        f"**Face {i}** → `{fid}`  \n"
                        f"{badge}  \n"
                        f"Similarity: `{sim:.3f}` · Detection conf: `{conf:.2f}`"
                    )
                else:
                    st.warning(
                        f"**Face {i}** · Detected (conf `{conf:.2f}`)  \n"
                        "Recognition unavailable or embedding extraction failed."
                    )

    st.caption(
        "ℹ️ **No ENTRY/EXIT events** are recorded for still images. "
        "The event model requires a multi-frame tracked video sequence. "
        "The visitor gallery (``visitors`` table) is updated normally."
    )


def _render_video_result(result: Dict[str, Any]) -> None:
    """Display the pipeline summary after a video upload is processed."""
    if not result.get("success"):
        st.error(f"Video processing failed: {result.get('error', 'Unknown error')}")
        stderr = result.get("stderr", "")
        if stderr:
            with st.expander("🔍 Error details (stderr)"):
                st.code(stderr[-3000:], language="log")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames Processed", result.get("frames_processed", 0))
    c2.metric("Visitors in this Video", result.get("unique_visitors", 0))
    c3.metric("ENTRY Events (this run)", result.get("entry_events", 0))
    c4.metric("EXIT Events (this run)", result.get("exit_events", 0))

    db_vis = result.get("total_db_visitors", 0)
    db_ent = result.get("total_db_entries", 0)
    db_ext = result.get("total_db_exits", 0)
    cumulative_info = ""
    if db_vis or db_ent or db_ext:
        cumulative_info = f"  |  💾 Cumulative Database: **{db_vis} unique visitors**, **{db_ent} entries**, **{db_ext} exits**"

    st.caption(
        f"⚡ Average throughput: **{result.get('fps', 0.0):.1f} FPS**  "
        f"· Video: `{Path(result.get('video_path', '?')).name}`"
        f"{cumulative_info}"
    )
    st.success("✅ Video processed successfully. Visitor and event tables below have been updated.")

    with st.expander("📜 Pipeline output log"):
        st.code(result.get("stdout", "")[-5000:], language="log")


# ---------------------------------------------------------------------------
# Upload section (image + video tabs)
# ---------------------------------------------------------------------------

def render_upload_section(config, db: Optional[DatabaseManager]) -> None:
    """Render the 'Upload Media for Face Tracking' panel."""
    with st.expander("📤 **Upload Media for Face Tracking**", expanded=True):
        img_tab, vid_tab = st.tabs(["📷 Image Detection", "🎬 Video Processing"])

        # ──────────────────────────────────────────────────────────────────
        # IMAGE TAB
        # ──────────────────────────────────────────────────────────────────
        with img_tab:
            st.markdown(
                "Upload a still image to run **YOLO face detection** and **ArcFace recognition**. "
                "New faces are registered in the visitor gallery. "
                "No ENTRY/EXIT events are created from a single image."
            )

            uploaded_img = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png"],
                key="img_uploader",
                help="Accepted: JPG · JPEG · PNG.  Max 50 MB.",
            )

            if uploaded_img is not None:
                size_kb = uploaded_img.size / 1024
                st.caption(f"📁 **{uploaded_img.name}** · {size_kb:.1f} KB")

                if size_kb > 50 * 1024:
                    st.error("File exceeds 50 MB. Upload a smaller image.")
                else:
                    col_preview, col_action = st.columns([3, 1])
                    with col_preview:
                        st.image(uploaded_img, caption="Preview", use_container_width=True)
                    with col_action:
                        st.write("")
                        st.write("")
                        btn_img = st.button(
                            "🔍 Process Image",
                            key="btn_process_img",
                            use_container_width=True,
                            help="Detect faces and register/recognize identities in the gallery",
                        )

                    if btn_img:
                        # Stable file identity: name + byte-size prevents reprocessing the same upload
                        file_id = f"{uploaded_img.name}|{uploaded_img.size}"
                        if st.session_state.get("_img_last_id") == file_id:
                            st.info("Already processed this image. Re-upload or select a different file to reprocess.")
                        else:
                            if db is None:
                                st.error("Database is not available. Cannot register visitors.")
                            else:
                                yolo_path = str(config.get_resolved_yolo_path())
                                detector = _get_detector(
                                    yolo_path,
                                    config.face_confidence_threshold,
                                    config.min_face_size,
                                )
                                if detector is None:
                                    st.error("Face detector could not be loaded. Check that model weights exist.")
                                else:
                                    recognizer = _get_recognizer(
                                        config.recognition_model_name,
                                        config.recognition_threshold,
                                        config.min_face_size,
                                    )
                                    with st.spinner("Running face detection and recognition…"):
                                        from src.upload_processor import process_image_upload
                                        from src.visitor_manager import VisitorManager

                                        vm = VisitorManager(
                                            db_manager=db,
                                            similarity_threshold=config.recognition_threshold,
                                        )
                                        scan_dir = config.project_root / "logs" / "image_scans"
                                        result = process_image_upload(
                                            file_bytes=uploaded_img.getvalue(),
                                            detector=detector,
                                            recognizer=recognizer,
                                            visitor_manager=vm,
                                            scan_dir=scan_dir,
                                        )

                                    st.session_state["img_result"] = result
                                    st.session_state["_img_last_id"] = file_id

            # Persist result display across Streamlit reruns
            if st.session_state.get("img_result"):
                st.markdown("---")
                st.markdown("##### 🔎 Detection Results")
                _render_image_result(st.session_state["img_result"])

        # ──────────────────────────────────────────────────────────────────
        # VIDEO TAB
        # ──────────────────────────────────────────────────────────────────
        with vid_tab:
            st.markdown(
                "Upload a video to run the **complete face tracking pipeline**: "
                "YOLO detection · ArcFace recognition · multi-face tracking · "
                "auto visitor registration · ENTRY/EXIT event recording · face crop archiving · database update."
            )

            uploaded_vid = st.file_uploader(
                "Choose a video file",
                type=["mp4", "avi", "mov"],
                key="vid_uploader",
                help="Accepted: MP4 · AVI · MOV.  Max 500 MB.  Processing time depends on video length and CPU speed.",
            )

            if uploaded_vid is not None:
                size_mb = uploaded_vid.size / (1024 * 1024)
                st.caption(f"📁 **{uploaded_vid.name}** · {size_mb:.1f} MB")

                if size_mb > 500:
                    st.error("File exceeds 500 MB limit. Upload a shorter clip.")
                else:
                    # Only embed player for small videos to avoid memory pressure
                    if size_mb < 50:
                        st.video(uploaded_vid)
                    else:
                        st.info(f"Video preview skipped for large files (>{size_mb:.0f} MB).")

                    btn_vid = st.button(
                        "▶️ Process Video",
                        key="btn_process_vid",
                        help="Save video and run the full face tracking pipeline (may take several minutes on CPU)",
                        disabled=bool(st.session_state.get("_video_processing")),
                    )

                    if btn_vid:
                        file_id = f"{uploaded_vid.name}|{uploaded_vid.size}"
                        already_done = (
                            st.session_state.get("_vid_last_id") == file_id
                            and st.session_state.get("video_result") is not None
                        )

                        if already_done:
                            st.info("This video was already processed. Results are shown below.")
                        else:
                            from src.upload_processor import save_uploaded_video, run_video_pipeline

                            upload_dir = config.project_root / "videos" / "uploads"
                            try:
                                with st.spinner("Saving video to disk…"):
                                    video_path = save_uploaded_video(
                                        file_bytes=uploaded_vid.getvalue(),
                                        original_name=uploaded_vid.name,
                                        upload_dir=upload_dir,
                                    )
                                st.info(
                                    f"✔ Saved as `{video_path.name}` in `videos/uploads/`. "
                                    "Starting pipeline…"
                                )
                            except Exception as exc:
                                st.error(f"Failed to save video to disk: {exc}")
                                video_path = None

                            if video_path is not None:
                                st.session_state["_video_processing"] = True
                                with st.spinner(
                                    "⚙️ Processing video — YOLO → ArcFace → Tracker → Events → DB.  "
                                    "This may take several minutes on CPU. Do not close this tab."
                                ):
                                    result = run_video_pipeline(video_path, config.project_root)

                                st.session_state["_video_processing"] = False
                                st.session_state["video_result"] = result
                                st.session_state["_vid_last_id"] = file_id
                                # Force a rerun so KPI metrics below reflect the updated database
                                st.rerun()

            # Persist result display across Streamlit reruns
            if st.session_state.get("video_result"):
                st.markdown("---")
                st.markdown("##### 📊 Pipeline Results")
                _render_video_result(st.session_state["video_result"])


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("👤 Katomaran Intelligent Face Tracker")
    st.caption("Auto-Registration · Re-Identification · Unique Visitor Analytics")

    config, db = load_app_data()

    if db is None:
        st.warning(
            f"Database not found at `{config.database_path}`. "
            "Run `main.py` or `scripts/initialize_database.py` to create it, "
            "or upload a video above to process one now."
        )
        # Still render upload section so the user can kick-start processing
        render_upload_section(config, db)
        st.stop()

    # ── Sidebar ─────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ System Control & Status")
    auto_refresh = st.sidebar.checkbox("Auto-refresh data (every 3 s)", value=False)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Video Source**: `{config.video_source}`")
    st.sidebar.markdown(f"**Recognition Model**: `{config.recognition_model_name}`")
    st.sidebar.markdown(f"**Matching Threshold**: `{config.recognition_threshold}`")
    st.sidebar.markdown(f"**Detection Skip**: `{config.detection_skip_frames} frames`")
    st.sidebar.markdown(f"**Grace Period**: `{config.exit_grace_frames} frames`")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Processing modes**\n\n"
        "- **Browser upload** → use the panel below\n"
        "- **Local file** → `main.py --source <path>`\n"
        "- **Live RTSP** → `main.py --source rtsp://…`"
    )

    # ── Upload section ───────────────────────────────────────────────────────
    render_upload_section(config, db)
    st.markdown("---")

    # ── KPI metric cards ─────────────────────────────────────────────────────
    visitor_count = db.get_unique_visitor_count()
    event_counts = db.get_event_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Unique Visitors", visitor_count)
    col2.metric("Total ENTRY Events", event_counts["ENTRY"])
    col3.metric("Total EXIT Events", event_counts["EXIT"])
    col4.metric(
        "Estimated Active In-Scene",
        max(0, event_counts["ENTRY"] - event_counts["EXIT"]),
    )
    st.markdown("---")

    # ── Main content tabs ────────────────────────────────────────────────────
    tab_visitors, tab_events, tab_gallery, tab_logs = st.tabs([
        "👥 Registered Visitors",
        "📋 Event Audit Trail",
        "🖼️ Face Image Gallery",
        "📄 System Events Log",
    ])

    with tab_visitors:
        st.subheader("Registered Persistent Identities")
        visitors = db.get_all_visitors()
        if visitors:
            st.dataframe(
                visitors,
                use_container_width=True,
                column_config={
                    "face_id":      "Face ID",
                    "first_seen":   "First Seen (UTC)",
                    "last_seen":    "Last Seen (UTC)",
                    "total_visits": "Total Visits",
                    "status":       "Current Status",
                },
            )
        else:
            st.info("No visitors registered yet. Upload an image or video to begin.")

    with tab_events:
        st.subheader("Recent Entry and Exit Events")
        events = db.get_events(limit=50)
        if events:
            st.dataframe(
                events,
                use_container_width=True,
                column_config={
                    "event_id":   "Event #",
                    "face_id":    "Face ID",
                    "event_type": "Event Type",
                    "timestamp":  "Timestamp (UTC)",
                    "track_id":   "Track ID",
                    "image_path": "Cropped Image Path",
                },
            )
        else:
            st.info("No events recorded yet.")

    with tab_gallery:
        st.subheader("Archived Face Event Crops")
        events_with_img = db.get_events(limit=24)
        if events_with_img:
            cols = st.columns(4)
            for idx, ev in enumerate(events_with_img):
                img_rel = ev.get("image_path")
                full_img_path = (config.project_root / img_rel) if img_rel else None
                with cols[idx % 4]:
                    if full_img_path and full_img_path.exists():
                        st.image(
                            str(full_img_path),
                            caption=f"{ev['face_id']} · {ev['event_type']} (#{ev['event_id']})",
                            use_container_width=True,
                        )
                    else:
                        st.text(
                            f"[{ev['face_id']}] {ev['event_type']}\n"
                            f"(path: {img_rel})"
                        )
        else:
            st.info("No face crop images in database. Process a video to generate event crops.")

    with tab_logs:
        st.subheader("Application Event Log  (logs/events.log)")
        log_file = config.get_resolved_log_path()
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-60:])
            st.code(tail, language="log")
        else:
            st.info(f"Log file not yet created at `{log_file}`.")


if __name__ == "__main__":
    main()
