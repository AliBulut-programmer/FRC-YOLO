import cv2
import numpy as np

# local imports


class FieldTransformer:
    def __init__(self, FIELD_SIZE, CSV_PATH):
        self.w, self.h = FIELD_SIZE
        self.src_p = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
        self.dst_p = np.array(
            [[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]], dtype=np.float32
        )
        self.H, _ = cv2.findHomography(self.src_p, self.dst_p)

    def transform_batch(self, detections):
        if detections is None or len(detections) == 0:
            return np.array([], dtype=np.int32)
        x1, x2, y2 = detections.xyxy[:, 0], detections.xyxy[:, 2], detections.xyxy[:, 3]

        bottom_centers = np.column_stack(((x1 + x2) / 2, y2))
        input_pts = bottom_centers.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(input_pts, self.H)
