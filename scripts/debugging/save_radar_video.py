import os
import sys
from collections import defaultdict
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import FIELD_SIZE, FOURCC, HOMOGRAPHY_CSV_PATH
from src.homography.mapping.homography_mapping import FieldTransformer
from src.homography.mapping.position_annotation import Radar


def save_radar(
    npz_path="outputs/detections/data.npz",
    output_path="outputs/radar_video.mp4",
):
    data = np.load(npz_path)
    frames = data["frame"]
    boxes = data["box"]
    tids = data["canonical_id"] if "canonical_id" in data else data["tracker_id"]
    fps = float(data["fps"]) if "fps" in data else 60.0

    by_frame = defaultdict(list)
    for i in range(len(frames)):
        by_frame[int(frames[i])].append(i)

    sorted_frames = sorted(by_frame.keys())
    if not sorted_frames:
        return

    transformer = FieldTransformer(FIELD_SIZE, HOMOGRAPHY_CSV_PATH)
    radar = Radar(FIELD_SIZE)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, FOURCC, fps, FIELD_SIZE)

    class DummyDetections:
        def __init__(self, xyxy):
            self.xyxy = xyxy

        def __len__(self):
            return len(self.xyxy)

    for frame_idx in sorted_frames:
        idxs = by_frame[frame_idx]
        dets = DummyDetections(boxes[idxs])
        positions = transformer.transform_batch(dets)
        frame_tids = tids[idxs]
        radar_frame = radar.map_positions(positions, frame_tids)
        out.write(radar_frame)

    out.release()

    if __name__ == "__main__":
        save_radar()
