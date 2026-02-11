import cv2
import numpy as np


class YOLO:
    def __init__(self, onnx_path, conf_thres=0.5, iou_thres=0.4, input_size=640):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_size = input_size

        # 加载模型
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # 预生成颜色 (确保每个颜色是3个值的元组)
        np.random.seed(42)
        self.colors = np.random.randint(0, 256, (100, 3), dtype=np.uint8)

    def preprocess(self, image):
        """图像预处理"""
        return cv2.dnn.blobFromImage(
            image, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False
        )

    def postprocess(self, image, outputs):
        """检测结果后处理"""
        h, w = image.shape[:2]
        outputs = np.squeeze(outputs).T

        # 提取预测结果
        cx, cy, bw, bh = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        cls_scores = outputs[:, 4:]

        # 计算置信度和类别
        confidences = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)

        # 过滤低置信度
        mask = confidences > self.conf_thres
        if not np.any(mask):
            return []

        # 坐标转换
        x = ((cx[mask] - bw[mask] / 2) * w / self.input_size).astype(int)
        y = ((cy[mask] - bh[mask] / 2) * h / self.input_size).astype(int)
        w = (bw[mask] * w / self.input_size).astype(int)
        h = (bh[mask] * h / self.input_size).astype(int)

        boxes = np.column_stack((x, y, w, h))
        scores = confidences[mask]
        class_ids = class_ids[mask]

        # NMS处理
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                                   self.conf_thres, self.iou_thres)

        if len(indices) == 0:
            return []

        indices = indices.flatten()
        return [(boxes[i], scores[i], class_ids[i]) for i in indices]

    def detect(self, image):
        """执行检测"""
        blob = self.preprocess(image)
        self.net.setInput(blob)
        outputs = self.net.forward()
        return self.postprocess(image, outputs)

    def draw(self, image, results):
        """绘制检测结果"""
        if not results:
            return image

        for box, score, cls_id in results:
            x, y, w, h = map(int, box)
            # 确保颜色是(B,G,R)格式的三元组
            color = tuple(map(int, self.colors[cls_id % 100]))

            # 绘制矩形框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            # 绘制类别和置信度
            label = f"{cls_id}:{score:.2f}"
            cv2.putText(image, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return image


if __name__ == "__main__":
    # 模型和图片路径
    model_path = "best.onnx"
    image_path = "654BCCE9E7F377FC478CAE7A267A1DAD.bmp"

    # 初始化模型
    yolo = YOLO(model_path, conf_thres=0.25, iou_thres=0.45)

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print("错误：无法读取图片")
        exit()

    # 执行检测
    results = yolo.detect(img)

    # 绘制结果
    out = yolo.draw(img.copy(), results)

    # 显示结果
    cv2.imshow("XD_YOLO_QQ1733617157", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()