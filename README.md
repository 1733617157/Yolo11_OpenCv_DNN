<!--
╔══════════════════════════════════════════════════════════════════════╗
║  DreamSeed 种梦计划 — AI创造者大赛  官方 README 模板                ║
║                                                                      ║
║  使用说明：                                                          ║
║  1. 将本模板放在参赛仓库根目录 README.md 的顶部                       ║
║  2. 头图使用 DreamField 官方公开活动图片地址                         ║
║  3. 请保留 DREAMFIELD_README_HEADER_START / END 标识                 ║
║  4. 分割线以下供创作者自由编写项目内容                               ║
╚══════════════════════════════════════════════════════════════════════╝
-->

<!-- DREAMFIELD_README_HEADER_START -->

<p align="center">
  <a href="https://www.dreamfield.top">
    <img src="https://www.dreamfield.top/dream-field/contest-readme/assets/dreamseed-readme-banner.png" alt="DreamSeed 种梦计划参赛作品" width="100%" />
  </a>
</p>

<!-- DREAMFIELD_README_HEADER_END -->


# Yolo11_OpenCv_DNN

注：通用于YoloV8 YoloV11 YoloV12 YoloV13 YoloV26

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
