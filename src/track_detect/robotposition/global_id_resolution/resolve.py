import cv2
import numpy as np
from collections import defaultdict

from src.config import VIDEO_PATH
from src.track_detect.robotposition.global_id_resolution.embedding import ViT
from src.track_detect.robotposition.global_id_resolution.stitcher import (
    TrackletStitcher,
)


# todo review this
def _cooccurring_pairs(frames, tracker_ids):
    ids_by_frame = defaultdict(set)
    for frame, tracker_id in zip(frames, tracker_ids):
        if tracker_id >= 0:
            ids_by_frame[int(frame)].add(int(tracker_id))

    pairs = set()
    for ids in ids_by_frame.values():
        ids = sorted(ids)
        for index, first in enumerate(ids):
            pairs.update((first, second) for second in ids[index + 1 :])
    return pairs


def embed(npz_path, video_path=VIDEO_PATH):

    data = np.load(npz_path)
    frames = data["frame"]
    boxes = data["box"]
    embs = data["emb"].astype(np.float32)

    vit = ViT()
    cap = cv2.VideoCapture(video_path)

    # npz saves data where each single detection gets its own row
    # by_frame creates a dictionary where
    #! keys are frame indices
    #! values are the row indices with that frame idx belongs to (detections in that frame)

    """
    | Row index (i) | `frames[i]` | `boxes[i]`      |
    | 0                     | 300           | [x1,y1,x2,y2] | 
    etc.
    """
    by_frame = defaultdict(list)
    for i in range(len(frames)):
        by_frame[int(frames[i])].append(i)
    # list:  sort the frames by order
    sorted_frames = sorted(by_frame.keys())
    if not sorted_frames:
        return

    # start from the first frame in the .npz file
    cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_frames[0])
    current_cap_frame = sorted_frames[0]  # init cap frame

    for frame_idx in sorted_frames:
        while current_cap_frame < frame_idx:
            # if frames are skipped in the .npz file for any reason, burn through the frames without detections
            cap.read()
            print("frame_skip")
            current_cap_frame += 1

        # ********************************* DEBUGGING ******************************** #

        ok, frame = cap.read()
        current_cap_frame += 1
        if not ok:
            continue

        frame = frame[:716]  # todo make this dynamic also fix in visualise

        for i in by_frame[frame_idx]:
            # i is the row index
            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)  # ensure x1,y1 are positive
            x2, y2 = min(frame.shape[1], x2), min(
                frame.shape[0], y2
            )  # ensure x2,y2 dont exceed frame limits
            if x2 <= x1 or y2 <= y1:  # if detections are faulty
                continue
            crop = frame[y1:y2, x1:x2]  # crop only the detection
            if current_cap_frame % 100 == 0:
                cv2.imwrite(
                    f"outputs/test_embed_crops/crop{current_cap_frame}.jpg", crop
                )

            embs[i] = vit.extract(crop).astype(np.float32)  # assign emb value

    cap.release()

    np.savez_compressed(
        npz_path,
        **{key: data[key] for key in data.files if key != "emb"},
        emb=embs.astype(np.float16),
    )  # data stays same unless its emb, if so assign extracted values


def stitch(npz_path, n_robots=6):
    """_summary_

    Args:
        npz_path (_type_): where the results will be saved
        n_robots (int, optional): actual number of robots in the scene. Defaults to 6.
    """
    # ------------------------- pull info from .npz file ------------------------- #
    data = np.load(npz_path)
    frames = data["frame"]  # frame idx from .npz file
    tracker_ids = data["tracker_id"]  # tracker_id data from .npz file
    embs = data["emb"].astype(np.float32)  # emb data from .npz file

    # get unique detections (excluding-1) and how many times they appear
    track_ids, track_counts = np.unique(
        tracker_ids[tracker_ids >= 0], return_counts=True
    )

    # turn unique detections and counts to dict
    tracklet_support = dict(zip(track_ids.tolist(), track_counts.tolist()))

    # -------------------- embedding calculation and matching -------------------- #
    stitcher = TrackletStitcher(n_robots=n_robots)  # initiate tracklet stitcher
    galleries = stitcher.build_galleries(tracker_ids, embs)  # create galleries
    mapping = stitcher.stitch(
        galleries,
        cooccurring_pairs=_cooccurring_pairs(frames, tracker_ids),
        tracklet_support=tracklet_support,
    )

    # ---------- create array giving corresponding canon ids of each tid --------- #
    canonical_ids = np.array(
        [mapping.get(int(tid), -1) for tid in tracker_ids],
        dtype=np.int16,
    )  # get the canonical ids of tid detections and make a matrix (-1 if not in mapping)

    # ********************************* DEBUGGING ******************************** #
    print("support:", sorted(tracklet_support.items(), key=lambda x: -x[1]))
    print("mapping:", mapping)
    print("unmapped:", [t for t in tracklet_support if t not in mapping])

    np.savez_compressed(
        npz_path,
        **{key: data[key] for key in data.files if key != "canonical_id"},
        canonical_id=canonical_ids,
    )
