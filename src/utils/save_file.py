import numpy as np

EMB_DIM = 384


class Recorder:
    def __init__(self, fps):
        self.rows = {
            k: []
            for k in [
                "frame",
                "box",
                "conf",
                "tracker_id",
                "cls",
                "clean",
                "alliance",
                "emb",
            ]
        }
        self.fps = fps

    def add(self, frame_idx, box, conf, tracker_id, cls, clean, alliance, emb):
        r = self.rows
        r["frame"].append(frame_idx)
        r["box"].append(box)
        r["conf"].append(conf)
        r["tracker_id"].append(tracker_id)
        r["cls"].append(cls)
        r["clean"].append(clean)
        r["alliance"].append(alliance)
        r["emb"].append(
            emb if emb is not None else np.full(EMB_DIM, np.nan, dtype=np.float32)
        )

    def save(self, path):
        r = self.rows
        np.savez_compressed(
            path,
            frame=np.array(r["frame"], dtype=np.int32),
            box=np.array(r["box"], dtype=np.float32).reshape(-1, 4),
            conf=np.array(r["conf"], dtype=np.float32),
            tracker_id=np.array(r["tracker_id"], dtype=np.int32),
            cls=np.array(r["cls"], dtype=np.int16),
            clean=np.array(r["clean"], dtype=bool),
            alliance=np.array(r["alliance"], dtype=np.float32),
            emb=np.array(r["emb"], dtype=np.float16).reshape(-1, EMB_DIM),
            fps=np.float32(self.fps),
        )
