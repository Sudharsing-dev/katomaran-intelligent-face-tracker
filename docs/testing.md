# Test Strategy, Scenarios, and Verification Report
## Katomaran Intelligent Face Tracker with Auto-Registration and Visitor Counting

---

## 1. Testing Methodology

The test suite is structured into isolated unit tests and pipeline integration tests using `pytest`.
Key testing principles:
- **Zero Heavy-Weight Dependency for Unit Tests**: Tests do not require downloading gigabytes of model weights or an active camera stream; synthetic frames and normalized embedding vectors are used.
- **Isolated Sandboxes**: Every test uses temporary filesystem directories (`tmp_path`) and isolated SQLite instances to guarantee complete test independence.
- **Strict Invariant Checking**: Focuses heavily on edge cases, such as fast repeated frames, missing frame grace periods, foreign key violations, and re-identification without count increments.

---

## 2. Test Scenarios and Expected Behavior

| Test Module | Test Case | Target Behavior & Invariant Checked |
| :--- | :--- | :--- |
| `test_config.py` | `test_default_config_validity` | Verifies default parameters parse and validate successfully. |
| `test_config.py` | `test_relative_path_resolution` | Ensures all output paths resolve relative to project root, not terminal CWD. |
| `test_config.py` | `test_invalid_skip_frames` | Rejects negative skip frame parameters with descriptive `ValueError`. |
| `test_config.py` | `test_invalid_exit_grace` | Rejects `exit_grace_frames < 1`. |
| `test_config.py` | `test_invalid_recognition_threshold` | Rejects thresholds outside $[0.0, 1.0]$. |
| `test_database.py` | `test_embedding_serialization_roundtrip` | Serializes 512-dim float32 vector to raw bytes (2048 bytes) and recovers without precision loss. |
| `test_database.py` | `test_embedding_nan_inf_rejection` | Prohibits NaN or Inf embeddings from entering database. |
| `test_database.py` | `test_foreign_key_enforcement` | Rejects event insertion for non-registered `face_id` (`PRAGMA foreign_keys = ON`). |
| `test_database.py` | `test_record_entry_and_exit_events` | Creates ENTRY and EXIT records; updates visitor status and timestamps. |
| `test_image_storage.py` | `test_crop_face_clipping` | Bounding boxes extending beyond image boundaries are safely clipped without crashing. |
| `test_image_storage.py` | `test_save_entry_and_exit_image` | Saves cropped images to `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`. |
| `test_image_storage.py` | `test_save_image_collision_prevention` | Rapid microsecond saves generate unique filenames without overwriting earlier evidence. |
| `test_visitor_manager.py` | `test_first_visitor_registration` | Unseen embedding registers as `FACE_001` and increments unique visitor count to 1. |
| `test_visitor_manager.py` | `test_returning_visitor_recognition` | Similar embedding ($\text{sim} > 0.80$) reuses `FACE_001` and does **not** increment count. |
| `test_visitor_manager.py` | `test_distinct_second_visitor_registration` | Orthogonal embedding registers as `FACE_002` and increments count to 2. |
| `test_visitor_manager.py` | `test_gallery_persistence_across_restarts` | Gallery is rebuilt accurately from database upon process restart. |
| `test_event_manager.py` | `test_single_entry_event_per_track` | Continuous tracking over 10 frames produces strictly **1** ENTRY event. |
| `test_event_manager.py` | `test_grace_period_and_exit_confirmation` | Missing frames within grace period trigger `EXIT_PENDING`; reappearance resumes `ACTIVE`; expiration confirms **1** EXIT event. |
| `test_event_manager.py` | `test_returning_visitor_new_lifecycle` | Person leaving and returning in a new track gets a new ENTRY event while reusing original Face ID. |
| `test_event_manager.py` | `test_multiple_independent_tracks` | Two distinct people tracked simultaneously without state collision or race conditions. |
| `test_visitor_count.py` | `test_unique_visitor_counting_guarantee` | End-to-end invariant test: 100 continuous frames and re-entry maintain strictly accurate visitor count. |

---

## 3. Running Automated Tests

To run the entire test suite:
```powershell
.\venv\Scripts\pytest -v tests/
```

To run with coverage reporting:
```powershell
.\venv\Scripts\pytest -v --cov=src tests/
```
