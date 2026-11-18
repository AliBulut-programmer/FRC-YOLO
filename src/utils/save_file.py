import numpy as np

EMB_DIM = 384
from pathlib import Path


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
        Path("outputs/detections").mkdir(parents=True, exist_ok=True)
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
            fps=np.float32(self.fps)
        )


def save_video_from_npz(
    npz_path: str = "outputs/detections/data.npz",
    output_path: str = "outputs/annotated_video.mp4",
    video_path: str | None = None,
    crop_height: int | None = 716,
    start_frame: int | None = None,
    end_frame: int | None = None,
    show_labels: bool = True,
    show_frame_num: bool = True,
    override_fps: float | None = None,
):
    """Renders and saves an annotated video using detection/tracking data from an .npz file."""
    from scripts.debugging.save_video import save_video
    if video_path is None:
        from src.config import VIDEO_PATH
        video_path = VIDEO_PATH

    return save_video(
        npz_path=npz_path,
        video_path=video_path,
        output_path=output_path,
        crop_height=crop_height,
        start_frame=start_frame,
        end_frame=end_frame,
        show_labels=show_labels,
        show_frame_num=show_frame_num,
        override_fps=override_fps,
    )

