# ----------------------------- external imports ----------------------------- #
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# ----------------------------- parameter imports ---------------------------- #
from src.config import (
    MODEL_PATH,
    CONF_THRESH,
    IMG_SIZE,
    VIDEO_PATH,
    DETECTION_IOU,
    START_FRAME_COUNT,
    END_FRAME_COUNT,
)

# ------------------------------- local imports ------------------------------ #
from src.track_detect.robotposition.agnostic_detection.detector import RobotDetection
from src.track_detect.robotposition.transient_tracking.tracker import RobotTracker
import src.utils as utils


def run(video_path=VIDEO_PATH, output_path="outputs/detections/data.npz"):
    """_summary_

    Args:
        video_path (_type_, optional): _description_. Defaults to VIDEO_PATH.
        output_path (str, optional): _description_. Defaults to "outputs/detections/data.npz".
    """
    # ****************************** INITIALIZATION ****************************** #
    detector = RobotDetection(MODEL_PATH, CONF_THRESH, IMG_SIZE, DETECTION_IOU)
    tracker = RobotTracker(fps=60)
    letterboxer = utils.LetterBoxer()
    recorder = utils.Recorder(60)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME_COUNT)  # set starting frame
    frame_count = START_FRAME_COUNT

    # ****************************** VIDEO READ LOOP ***************************** #
    while cap.isOpened():

        ok, frame = cap.read()
        if not ok:
            break

        frame = frame[:716]
        # todo make this parameter dynamic,in resolve and visualise too

        letterboxed = letterboxer.letterbox(frame, IMG_SIZE)
        # --------------------------------- detection -------------------------------- #
        detections = detector.predict(letterboxed)
        # --------------------------------- tracking --------------------------------- #
        detections = tracker.track(frame=letterboxed, detections=detections)

        letterboxer.scale_detections(detections)

        for i in range(len(detections)):
            recorder.add(
                frame_idx=frame_count,
                box=np.float32(detections.xyxy[i]),
                conf=float(detections.confidence[i]),
                tracker_id=int(detections.tracker_id[i]),
                cls=int(detections.class_id[i]),
                clean=True,
                alliance=None,
                emb=None,
            )

        print(frame_count)
        if frame_count == END_FRAME_COUNT:
            break
        frame_count += 1

    cap.release()
    recorder.save(output_path)
