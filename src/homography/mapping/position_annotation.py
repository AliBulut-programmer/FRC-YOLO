import cv2
import numpy as np


class Radar:
    def __init__(self, FIELD_SIZE):
        self.canvas_w, self.canvas_h = FIELD_SIZE
        self.blank_frame = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

    def map_positions(self, positions, tracker_ids):
        self.blank_frame.fill(255)
        if positions is None or len(positions) == 0:
            return self.blank_frame

        for i in range(len(positions)):
            tracker_id = tracker_ids[i]
            color = (
                int((tracker_id * 37) % 255),
                int((tracker_id * 83) % 255),
                int((tracker_id * 131) % 255),
            )
            x = int(positions[i, 0, 0])
            y = int(positions[i, 0, 1])
            cv2.circle(self.blank_frame, (x, y), radius=8, color=color, thickness=4)
        return self.blank_frame
