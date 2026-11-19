import supervision as sv
from trackers import OCSORTTracker
import numpy as np
import warnings
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
from src.config import (
    TRACK_THRESH,
    MIN_CONSECUTIVE_FRAME,
    LOST_TRACK_BUFFER,
    INERTIA,
    MIN_IOU_TRACKER,
)
import math


class RobotTracker:
    def __init__(self, fps=60):
        self.tracker = OCSORTTracker(
            high_conf_det_threshold=TRACK_THRESH,
            frame_rate=fps,
            lost_track_buffer=LOST_TRACK_BUFFER,
            minimum_consecutive_frames=MIN_CONSECUTIVE_FRAME,
            direction_consistency_weight=INERTIA,
            minimum_iou_threshold=MIN_IOU_TRACKER,
        )

    def track(self, frame, detections):
        if not detections:
            detections = sv.Detections.empty()
        else:
            detections = sv.Detections.from_ultralytics(detections)
        result = self.tracker.update(detections)
        return result
