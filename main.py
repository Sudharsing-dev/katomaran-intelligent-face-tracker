"""Main entry point and orchestrator for Katomaran Intelligent Face Tracker.

Coordinates:
1. Configuration loading and validation
2. Structured logging to logs/events.log
3. SQLite database persistence
4. YOLO face detection
5. InsightFace / ArcFace identity recognition & auto-registration
6. Multi-face tracking and state transitions
7. ENTRY and EXIT event lifecycle management
8. Cropped face image archiving
9. Real-time overlay visualization and performance monitoring
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config
from src.database import DatabaseManager
from src.event_manager import EventManager
from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.face_tracker import FaceTracker
from src.image_storage import ImageStorageManager, crop_face
from src.logger import get_logger, setup_logger
from src.performance_monitor import PerformanceMonitor
from src.video_reader import VideoReader
from src.visitor_manager import VisitorManager


def draw_hud_and_boxes(
    frame: np.ndarray,
    active_tracks: list,
    fps: float,
    unique_visitors: int,
    total_entries: int,
    total_exits: int,
) -> np.ndarray:
    """Draw bounding boxes, track IDs, Face IDs, and a heads-up display on the frame."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw semi-transparent top header bar
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Header HUD text
    hud_text = (
        f"FPS: {fps:.1f} | Unique Visitors: {unique_visitors} | "
        f"Entries: {total_entries} | Exits: {total_exits} | Active Tracks: {len(active_tracks)}"
    )
    cv2.putText(
        annotated,
        hud_text,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Draw face bounding boxes
    for track in active_tracks:
        x1, y1, x2, y2 = [int(round(c)) for c in track.bbox]
        fid = track.face_id or f"Track #{track.track_id}"
        is_recognized = track.face_id is not None

        # Color: Bright Green for confirmed identity, Orange/Cyan for pending identification
        color = (0, 255, 0) if is_recognized else (255, 180, 0)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{fid} (T#{track.track_id})"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y1 = max(0, y1 - lh - 8)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + lw + 8, label_y1 + lh + 8), color, -1)

        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y1 + lh + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def run_pipeline(config: AppConfig, headless: bool = False) -> int:
    """Execute complete face detection, tracking, recognition, and visitor counting pipeline."""
    logger = setup_logger(
        log_path=config.get_resolved_log_path(),
        log_level=config.log_level,
    )
    logger.info("=" * 70)
    logger.info("Starting Katomaran Intelligent Face Tracker application...")
    logger.info(f"Loaded configuration: source='{config.video_source}', skip_frames={config.detection_skip_frames}")

    # 1. Initialize SQLite Database
    db_path = config.get_resolved_database_path()
    logger.info(f"Initializing database at: {db_path}")
    db_manager = DatabaseManager(db_path)

    # 2. Initialize Image Storage
    storage_manager = ImageStorageManager(
        project_root=config.project_root,
        entry_dir=config.entry_image_dir,
        exit_dir=config.exit_image_dir,
    )

    # 3. Initialize Visitor Manager
    visitor_manager = VisitorManager(
        db_manager=db_manager,
        similarity_threshold=config.recognition_threshold,
    )

    # 4. Initialize Tracker
    tracker = FaceTracker(
        max_missing_frames=config.exit_grace_frames,
        detection_skip_frames=config.detection_skip_frames,
    )

    # 5. Initialize Event Manager
    event_manager = EventManager(
        db_manager=db_manager,
        image_storage=storage_manager,
        exit_grace_frames=config.exit_grace_frames,
        end_of_stream_policy=config.end_of_stream_policy,
    )

    # 6. Initialize Performance Monitor
    perf_monitor = PerformanceMonitor()

    # 7. Initialize AI Models (Face Detector & Face Recognizer)
    detector = None
    yolo_path = config.get_resolved_yolo_path()
    try:
        detector = FaceDetector(
            model_path=yolo_path,
            confidence_threshold=config.face_confidence_threshold,
            min_face_size=config.min_face_size,
        )
    except Exception as e:
        logger.error(
            f"Face detector initialization failed: {e}.\n"
            "If weights are not downloaded, run: python -m src.face_detector --download"
        )
        # Continue with warning to allow test runs or graceful failure
        print(f"\n[CRITICAL WARNING] Face detector could not be loaded: {e}\n")

    recognizer = None
    try:
        recognizer = FaceRecognizer(
            model_name=config.recognition_model_name,
            similarity_threshold=config.recognition_threshold,
            min_face_size=config.min_face_size,
        )
    except Exception as e:
        logger.warning(f"Face recognizer initialization failed: {e}")

    # 8. Open Video Input
    resolved_source = config.get_resolved_video_source()
    video_reader = VideoReader(
        source=resolved_source,
        reconnect_delay_sec=config.rtsp_reconnect_delay_sec,
        max_reconnect_attempts=config.rtsp_max_reconnect_attempts,
        target_width=config.frame_resize_width,
    )

    # Guard: abort cleanly when an RTSP camera could not be connected.
    # We do this BEFORE opening any windows or touching the database to avoid
    # creating spurious records or showing a blank display window.
    if not video_reader.connected:
        if video_reader.is_rtsp:
            # Error message already printed inside _initial_rtsp_connect()
            logger.error(
                "Pipeline aborted: RTSP source could not be connected. "
                "No visitor data has been modified."
            )
            print(
                "\n[ABORTED] Pipeline did not start — RTSP camera unreachable.\n"
                "          No visitor records, event records, or face images were created.\n"
                "          Check the URL, network connectivity, and camera credentials.\n"
            )
        else:
            logger.error(
                f"Pipeline aborted: video source '{video_reader.get_sanitized_source_name()}' "
                "could not be opened. Verify the file path."
            )
            print(
                f"\n[ERROR] Could not open video source: "
                f"{video_reader.get_sanitized_source_name()}\n"
                "        Verify the file path and try again.\n"
            )
        video_reader.release()
        return 1

    display_enabled = config.display_video and not headless
    window_name = "Katomaran Intelligent Face Tracker"
    if display_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 65)
    print("  KATOMARAN INTELLIGENT FACE TRACKER IS RUNNING")
    print(f"  Source: {video_reader.get_sanitized_source_name()}")
    print(f"  Initial Registered Visitors: {visitor_manager.unique_visitor_count}")
    print("  Press 'q' or 'ESC' in video window to exit safely.")
    print("=" * 65 + "\n")

    try:
        for frame_idx, frame in video_reader.frames():
            perf_monitor.start_frame()

            # Determine whether this is a full YOLO detection frame
            is_detection_frame = (frame_idx % (config.detection_skip_frames + 1) == 0)

            detections = []
            if is_detection_frame and detector is not None:
                t0 = time.time()
                detections = detector.detect(frame)
                perf_monitor.record_detection(time.time() - t0)

            # Update Tracker with detections or motion predictions
            t_track = time.time()
            active_tracks = tracker.update(detections, is_detection_frame=is_detection_frame)
            perf_monitor.record_tracking(time.time() - t_track)

            active_track_ids = [t.track_id for t in active_tracks]

            # Process identities for active tracks
            for track in active_tracks:
                # If track lacks a Face ID, generate embedding and register/recognize
                if not track.face_id and recognizer is not None:
                    crop = crop_face(frame, track.bbox, margin_ratio=0.30)
                    if crop is not None:
                        t_rec = time.time()
                        emb = recognizer.extract_embedding(crop)
                        perf_monitor.record_recognition(time.time() - t_rec)

                        if emb is not None:
                            face_id, is_new, sim_score = visitor_manager.register_or_recognize(emb)
                            track.face_id = face_id
                            tracker.associate_face_id(track.track_id, face_id)
                            # Notify event manager of identification
                            event_manager.update_track_detection(
                                track_id=track.track_id,
                                bbox=track.bbox,
                                frame=frame,
                                face_id=face_id,
                                similarity_score=sim_score,
                                is_new_visitor=is_new,
                            )
                            continue

                # Update track detection in event manager
                event_manager.update_track_detection(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    frame=frame,
                    face_id=track.face_id,
                )

            # Check for disappeared tracks that exceed grace period -> confirm EXIT
            event_manager.process_missing_tracks(active_track_ids)

            # Visualization
            if display_enabled:
                event_counts = db_manager.get_event_counts()
                annotated = draw_hud_and_boxes(
                    frame=frame,
                    active_tracks=active_tracks,
                    fps=perf_monitor.current_fps,
                    unique_visitors=visitor_manager.unique_visitor_count,
                    total_entries=event_counts["ENTRY"],
                    total_exits=event_counts["EXIT"],
                )
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    logger.info("User requested exit via keyboard ('q'/ESC).")
                    break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user (KeyboardInterrupt).")
    finally:
        # Handle End-of-Stream policy
        event_manager.handle_end_of_stream()
        video_reader.release()
        if display_enabled:
            cv2.destroyAllWindows()

    # Final summary report
    summary = perf_monitor.get_summary()
    session_visitors = event_manager.session_visitor_count
    session_entries = event_manager.session_entry_count
    session_exits = event_manager.session_exit_count
    total_db_visitors = visitor_manager.unique_visitor_count
    total_db_events = db_manager.get_event_counts()

    logger.info("=" * 70)
    logger.info("FINAL PROCESSING STATISTICS:")
    logger.info(f"  Total Frames Processed : {summary['total_frames']}")
    logger.info(f"  Elapsed Duration       : {summary['elapsed_seconds']} s")
    logger.info(f"  Overall Throughput     : {summary['overall_fps']} FPS")
    logger.info(f"  Session Unique Visitors: {session_visitors}")
    logger.info(f"  Session ENTRY Events   : {session_entries}")
    logger.info(f"  Session EXIT Events    : {session_exits}")
    logger.info(f"  Total DB Visitors      : {total_db_visitors}")
    logger.info(f"  Total DB ENTRY Events  : {total_db_events['ENTRY']}")
    logger.info(f"  Total DB EXIT Events   : {total_db_events['EXIT']}")
    logger.info("Katomaran Face Tracker shutdown complete.")
    logger.info("=" * 70)

    print("\n" + "=" * 65)
    print("  PROCESSING COMPLETED SUMMARY")
    print("=" * 65)
    print(f"  Total Frames Processed : {summary['total_frames']}")
    print(f"  Overall Average FPS    : {summary['overall_fps']}")
    print(f"  Session Unique Visitors: {session_visitors}")
    print(f"  Session ENTRY Events   : {session_entries}")
    print(f"  Session EXIT Events    : {session_exits}")
    print(f"  Total Unique Visitors  : {total_db_visitors}")
    print(f"  Confirmed ENTRY Events : {total_db_events['ENTRY']}")
    print(f"  Confirmed EXIT Events  : {total_db_events['EXIT']}")
    print("=" * 65 + "\n")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Katomaran Intelligent Face Tracker with Auto-Registration and Visitor Counting."
    )
    parser.add_argument("--source", type=str, default=None, help="Video file path or RTSP stream URL.")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json.")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window.")
    parser.add_argument("--skip-frames", type=int, default=None, help="Number of frames to skip between detections.")
    parser.add_argument("--threshold", type=float, default=None, help="Recognition cosine similarity threshold.")
    args = parser.parse_args()

    # Prepare command-line overrides
    overrides = {}
    if args.source:
        overrides["video_source"] = args.source
    if args.skip_frames is not None:
        overrides["detection_skip_frames"] = args.skip_frames
    if args.threshold is not None:
        overrides["recognition_threshold"] = args.threshold
    if args.no_display:
        overrides["display_video"] = False

    try:
        config = load_config(config_path=args.config, overrides=overrides)
    except Exception as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1

    return run_pipeline(config=config, headless=args.no_display)


if __name__ == "__main__":
    sys.exit(main())
