import numpy as np
import pandas as pd


def calc_bbox(
    frame_count,
    detections,
    init_frame_count,
    BBOX_THRESH_CALC_FRAME,
    avg_bbox: list,
    AVG_BBOX_SCALE_FACTOR=0.5,
):

    for i, detection in enumerate(detections.boxes.xywh.cpu().numpy()):
        x, y, w, h = detection
        w, h = int(w), int(h)

        avg_bbox.append((w, h))

    if frame_count < BBOX_THRESH_CALC_FRAME + init_frame_count:
        use_bbox = False

    else:
        df = pd.DataFrame(avg_bbox, columns=["w", "h"])

        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)

        iqr = q3 - q1
        mask = ((df < q1 - 1.5 * iqr) | (df > q3 + 1.5 * iqr)).any(axis=1)

        df = df[~mask].to_numpy()

        w_sum, h_sum = 0, 0
        for i, size in enumerate(df):
            w_sum += size[0]
            h_sum += size[1]
        leng = len(df)
        avg_w, avg_h = (w_sum // leng), (h_sum // leng)
        avg_w = int(avg_w * AVG_BBOX_SCALE_FACTOR)
        avg_h = int(avg_h * AVG_BBOX_SCALE_FACTOR)

        avg_bbox = np.array((avg_w, avg_h))
        use_bbox = True
    return avg_bbox, use_bbox


if __name__ == "__main__":
    calc_bbox
