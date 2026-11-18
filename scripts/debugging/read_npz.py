import numpy as np

data = np.load("outputs/detections/data.npz")

print(
    "Frames recorded:", data["frame"][:10]
)  # Should show frame numbers [0, 0, 0, 1, 1...]
print("Boxes shape:", data["box"].shape)  # Should be (N, 4)
print("Sample box:", data["box"][0])  # Should show [x1, y1, x2, y2]
print("Tracker IDs:", data["tracker_id"][:10])  # Should show integer IDs
if "canonical_id" in data:
    print("Canonical robot IDs:", np.unique(data["canonical_id"]))
else:
    print("Canonical robot IDs: not found; run the embedding and stitching steps")
print("Confidences:", data["conf"][:10])  # Should show confidence floats
