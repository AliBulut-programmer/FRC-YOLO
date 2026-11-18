import cv2
from src.config import LETTERBOX_COLOR


class LetterBoxer:
    def __init__(self):
        pass

    def letterbox(self, frame, output_size):
        self.in_h, self.in_w = frame.shape[:2]
        self.ratio = min(output_size / self.in_h, output_size / self.in_w)
        resized = cv2.resize(
            frame, None, fx=self.ratio, fy=self.ratio, interpolation=cv2.INTER_LINEAR
        )

        self.pad_w = output_size - self.in_w * self.ratio
        self.pad_h = output_size - self.in_h * self.ratio

        top = bottom = int(self.pad_h // 2)
        left = right = int(self.pad_w // 2)

        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=LETTERBOX_COLOR,
        )
        return letterboxed

    def remove_letterbox(self, frame):
        out_h1 = int(self.pad_h // 2)
        out_w1 = int(self.pad_w // 2)

        # The correct slice end is the starting offset plus the actual resized dimensions
        out_h2 = out_h1 + int(round(self.in_h * self.ratio))
        out_w2 = out_w1 + int(round(self.in_w * self.ratio))

        return frame[out_h1:out_h2, out_w1:out_w2]

    def scale_detections(self, detections):
        detections.xyxy[:, 0] = (
            detections.xyxy[:, 0] - self.pad_w // 2
        ) // self.ratio  # x1
        detections.xyxy[:, 1] = (
            detections.xyxy[:, 1] - self.pad_h // 2
        ) // self.ratio  # y1
        detections.xyxy[:, 2] = (
            detections.xyxy[:, 2] - self.pad_w // 2
        ) // self.ratio  # x2
        detections.xyxy[:, 3] = (
            detections.xyxy[:, 3] - self.pad_h // 2
        ) // self.ratio  # y2
