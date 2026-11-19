import numpy as np


def load(p):
    d = np.load(p)
    return {k: d[k] for k in d.files}


a, b = load("outputs/detections/short.npz"), load("outputs/detections/long.npz")
last = a["frame"].max()


def window(d):
    m = d["frame"] <= last
    return {k: d[k][m] for k in ("frame", "box", "conf", "tracker_id")}


wa, wb = window(a), window(b)
print("same row count:", len(wa["frame"]) == len(wb["frame"]))
if len(wa["frame"]) == len(wb["frame"]):
    print("boxes identical:", np.allclose(wa["box"], wb["box"], atol=1e-3))
    print("tracker_id identical:", np.array_equal(wa["tracker_id"], wb["tracker_id"]))
