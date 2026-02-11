# Yolo11_OpenCv_DNN

基于 **OpenCV DNN** 的 **YOLO11** 推理示例项目，使用 **ONNX 模型**，无需 PyTorch / TensorRT，适合  
**部署、离线推理、嵌入式 / Windows / Linux 环境**。

---

## ✨ 特性

- ✅ 基于 OpenCV `dnn` 模块
- ✅ 支持 YOLO11 ONNX 模型
- ✅ 纯 Python 推理，无需 PyTorch
- ✅ 支持图片 / 视频 / 摄像头
- ✅ 方便移植到 C++ / 嵌入式环境
- ✅ 适合生产部署与二次开发

---

## 📦 环境依赖

- Python ≥ 3.8
- OpenCV ≥ 4.8（必须包含 `dnn` 模块）
- NumPy

安装依赖：
```bash
pip install opencv-python numpy
