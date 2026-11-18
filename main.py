from src.track_detect.track_detect import run as run_track_detect
from src.track_detect.robotposition.global_id_resolution.resolve import embed, stitch


def main():
    npz_path = "outputs/detections/data.npz"
    run_track_detect(output_path=npz_path)
    embed(npz_path=npz_path)
    stitch(npz_path=npz_path)


if __name__ == "__main__":
    main()
