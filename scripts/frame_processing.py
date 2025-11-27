import cv2
import torch
import numpy as np
from frame_loader import FrameLoader
from model_inferences import YOLOv8NInference
from transform_factory import get_transform


class FrameProcessing:
    def __init__(self, video_path, model_key, device="cuda"):
        self.face_model = YOLOv8NInference()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.transform = get_transform(model_key)
        
        self.frames = FrameLoader(video_path)

    def process(self):
        for frame, idx in self.frames:
            yolo_result = self.face_model.predict(frame)
            face_img, bbox = self.face_crop(frame, yolo_result)
            face_tensor = self.to_tensor(face_img)

            yield idx, frame, face_tensor, bbox

    def face_crop(self, frame, yolo_result):
        boxes = yolo_result[0].boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            return None, None

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = boxes[np.argmax(areas)].astype(int)
        x1, y1, x2, y2 = box

        # Thêm margin nhẹ để tránh crop sát mặt
        h, w = frame.shape[:2]
        margin = 20
        x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
        x2, y2 = min(w, x2 + margin), min(h, y2 + margin)

        return frame[y1:y2, x1:x2], box

    def to_tensor(self, face_img):
        if face_img is None or face_img.size == 0:
            return None
        aug = self.transform(image=cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        return aug["image"].unsqueeze(0).to(self.device)
