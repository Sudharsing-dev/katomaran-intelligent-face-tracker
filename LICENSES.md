# Third-Party Licenses & Model Acknowledgments

This document lists the open-source libraries, computer vision architectures, and pre-trained models used or supported in the **Katomaran Intelligent Face Tracker**.

---

## 1. Ultralytics YOLO (Face Detection)
- **Component**: YOLOv8 / YOLOv11 Face Detection models (e.g., `yolov8n-face.pt`)
- **Repository**: [Ultralytics](https://github.com/ultralytics/ultralytics) / [yolov8-face](https://github.com/derronqi/yolov8-face)
- **License**: AGPL-3.0 / Enterprise License (Ultralytics). Ensure compliance with AGPL-3.0 or commercial licensing when distributing derived binaries or web services.

---

## 2. InsightFace & ArcFace (Face Recognition & Embeddings)
- **Component**: InsightFace Python Library, ArcFace (Additive Angular Margin Loss for Deep Face Recognition)
- **Repository**: [deepinsight/insightface](https://github.com/deepinsight/insightface)
- **License**: MIT License for the codebase. Note: Pre-trained models (`buffalo_l`, `buffalo_sc`, `antelopev2`) are provided under non-commercial research licenses by DeepInsight unless custom trained.

---

## 3. ByteTrack / SORT (Object & Face Tracking)
- **Component**: ByteTrack multi-object tracking algorithm
- **Repository**: [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)
- **License**: MIT License

---

## 4. OpenCV (Open Source Computer Vision Library)
- **Component**: Image processing, frame capture, video decoding (`opencv-python`)
- **License**: Apache 2.0 License

---

## 5. NumPy & SciPy
- **Component**: Numerical computations, vector normalization, cosine similarity
- **License**: BSD 3-Clause License

---

## 6. PyTorch & ONNX Runtime
- **Component**: Deep learning inference engines
- **License**: PyTorch (Modified BSD License), ONNX Runtime (MIT License)

---

## 7. Katomaran Project Code
- **License**: Developed for Katomaran Hackathon. All application orchestration logic, event management, SQLite persistence, and tracking state machine are provided under the MIT License.
