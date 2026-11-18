# ----------------------------- external imports ----------------------------- #
import sys
import os
import cv2
import numpy as np
from collections import defaultdict

# ------------------------------- parameter imports ------------------------------ #
# append root folder path for local parameter imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import VIDEO_PATH, FONT, FONT_SCALE, THICKNESS


def id_color(tracker_id):
    return (
        int((tracker_id * 37) % 255),
        int((tracker_id * 83) % 255),
        int((tracker_id * 131) % 255),
    )


def run(npz_path="outputs/detections/data.npz", video_path=VIDEO_PATH):
    """llll

    Args:
        npz_path (str, optional): _description_. Defaults to "outputs/detections/data.npz".
        video_path (_type_, optional): _description_. Defaults to VIDEO_PATH.
    """
    data = np.load(npz_path)
    frames = data["frame"]
    boxes = data["box"]
    confs = data["conf"]
    has_canonical_ids = "canonical_id" in data
    tids = data["canonical_id"] if has_canonical_ids else data["tracker_id"]
    id_label = "robot" if has_canonical_ids else "tracker"

    # npz saves data where each single detection gets its own row
    # by_frame creates a dictionary where
    #! keys are frame indices
    #! values are the row indices with that frame idx belongs to (detections in that frame)
    by_frame = defaultdict(list)
    for i in range(len(frames)):
        by_frame[int(frames[i])].append(i)

    cap = cv2.VideoCapture(video_path)

    sorted_frames = sorted(by_frame.keys())
    current_cap_frame = sorted(by_frame.keys())[0]
    if not sorted_frames:
        print("no frames in .npz file")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_frames[0])

    # iterate through frame keys in ascending order
    for frame_idx in sorted_frames:
        # burn through the frames before the intended frame // more secure and doesnt freeze as much compared to cap.set every frame
        while current_cap_frame < frame_idx:
            cap.read()
            current_cap_frame += 1

        ok, frame = cap.read()
        current_cap_frame += 1

        if not ok:
            continue

        frame = frame[:716]  # todo make this dynamic also fix in resolve,

        # for every detection in current frame (i row index for data.npz)
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
        cv2.imshow("debug", frame)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("debug", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
