import sys
import os
import cv2
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import VIDEO_PATH, FONT, FONT_SCALE, THICKNESS, FOURCC


def id_color(tracker_id):
    return (
        int((tracker_id * 37) % 255),
        int((tracker_id * 83) % 255),
        int((tracker_id * 131) % 255),
    )


def run(
    npz_path="outputs/detections/data.npz",
    video_path=VIDEO_PATH,
    output_path="outputs/annotated_video.mp4",
):
    data = np.load(npz_path)
    frames = data["frame"]
    boxes = data["box"]
    confs = data["conf"]
    has_canonical_ids = "canonical_id" in data
    tids = data["canonical_id"] if has_canonical_ids else data["tracker_id"]
    id_label = "robot" if has_canonical_ids else "tracker"

    by_frame = defaultdict(list)
    for i in range(len(frames)):
        by_frame[int(frames[i])].append(i)

    sorted_frames = sorted(by_frame.keys())
    if not sorted_frames:
        print("no frames in .npz file")
        return

    cap = cv2.VideoCapture(video_path)
    fps = float(data["fps"]) if "fps" in data else cap.get(cv2.CAP_PROP_FPS)

    current_cap_frame = sorted_frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_frames[0])

    out = None

    for frame_idx in sorted_frames:
        while current_cap_frame < frame_idx:
            cap.read()
            current_cap_frame += 1

        ok, frame = cap.read()
        current_cap_frame += 1
        if not ok:
            continue

        frame = frame[:716]

        if out is None:
            h, w = frame.shape[:2]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, FOURCC, fps, (w, h))

        for i in by_frame[frame_idx]:
            x1, y1, x2, y2 = boxes[i].astype(int)
            tid = int(tids[i])
            conf = float(confs[i])
            color = id_color(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
            cv2.putText(
                frame,
                f"{conf:.2f} | {id_label} id:{tid}",
                (x1, y1 - 8),
                FONT,
                FONT_SCALE,
                color,
                THICKNESS,
            )

        cv2.putText(
            frame,
            f"frame:{frame_idx}",
            (10, 20),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            THICKNESS,
        )

        out.write(frame)
        print(frame_idx)
    cap.release()
    if out is not None:
        out.release()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    run()
