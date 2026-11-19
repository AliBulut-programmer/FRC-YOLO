"""
Bumper OCR test.

For every image in CROP_DIR (robot crops, ideally cut from the ORIGINAL frame,
not the letterboxed one):
  1. find the bumper via HSV colour masking (red / blue)
  2. crop + deskew it
  3. preprocess (upscale, CLAHE, a few binarised variants)
  4. run EasyOCR restricted to digits
  5. fuzzy-match / vote the reads against the TEAMS list

Debug images go to OUT_DIR so you can see what each stage did.

pip install easyocr opencv-python numpy
"""

import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import easyocr
import numpy as np

# ------------------------------- settings ---------------------------------- #
CROP_DIR = "outputs/crops"  # folder with robot crops
OUT_DIR = "outputs/bumper_test"  # debug output
TEAMS = [
    "868",
    "2046",
    "2910",
    "4065",
    "4414",
    "1323",
]  # edit: the 6 teams in the match

BUMPER_REGION = (
    0.35,
    1.0,
)  # only search this vertical fraction of the crop (bumpers sit low)
TARGET_HEIGHT = 128  # upscale bumper crop to this height before OCR
MIN_BUMPER_AREA_FRAC = (
    0.02  # ignore colour blobs smaller than this fraction of the crop
)
MIN_SIM = 0.6  # minimum string similarity to count as a vote for a team

# HSV ranges (OpenCV: H 0-179). Tune these on your footage.
RED_RANGES = [((0, 100, 60), (10, 255, 255)), ((170, 100, 60), (179, 255, 255))]
BLUE_RANGES = [((100, 100, 50), (130, 255, 255))]


# ------------------------------ localisation ------------------------------- #
def color_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def find_bumper(img):
    """Returns (bumper_crop, alliance, debug_mask) or (None, None, mask)."""
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # restrict to lower part of the crop
    region = np.zeros((h, w), dtype=np.uint8)
    region[int(h * BUMPER_REGION[0]) : int(h * BUMPER_REGION[1]), :] = 255

    masks = {
        "red": color_mask(hsv, RED_RANGES) & region,
        "blue": color_mask(hsv, BLUE_RANGES) & region,
    }
    alliance = max(masks, key=lambda k: int(masks[k].sum()))
    mask = masks[alliance]

    # close so the white digits (holes in the mask) get filled in
    k = max(5, w // 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, max(3, k // 2)))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, closed
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_BUMPER_AREA_FRAC * h * w:
        return None, None, closed

    # deskew using the min-area rectangle of the bumper blob
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
    if rw < rh:  # make the long side horizontal
        angle += 90
    if angle > 45:
        angle -= 90
    if angle < -45:
        angle += 90
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(
        img, rot, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    rot_mask = cv2.warpAffine(closed, rot, (w, h), flags=cv2.INTER_NEAREST)

    ys, xs = np.where(rot_mask > 0)
    if len(xs) == 0:
        return None, None, closed
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    padx, pady = int((x2 - x1) * 0.05), int((y2 - y1) * 0.1)
    x1, y1 = max(0, x1 - padx), max(0, y1 - pady)
    x2, y2 = min(w, x2 + padx), min(h, y2 + pady)
    return rotated[y1:y2, x1:x2], alliance, closed


# ------------------------------ preprocessing ------------------------------ #
def preprocess_variants(bumper):
    """Returns {name: single-channel image} to feed to OCR."""
    scale = TARGET_HEIGHT / bumper.shape[0]
    big = cv2.resize(bumper, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (1, 1), 0)

    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    # digits are white on a dark-ish coloured bumper, so also try the "white-ness" channel
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 150), (179, 80, 255))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    variants = {"gray": gray, "clahe": clahe, "adaptive": adaptive}
    # pad so digits touching the edge are still readable
    return {
        k: cv2.copyMakeBorder(
            v, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=int(np.median(v))
        )
        for k, v in variants.items()
    }


# --------------------------------- matching -------------------------------- #
def similarity(text, team):
    if text == team:
        return 1.0
    sim = SequenceMatcher(None, text, team).ratio()
    # partial read of a real team number (e.g. "167" for "1678") is still evidence
    if len(text) >= 2 and text in team:
        sim = max(sim, 0.6 + 0.4 * len(text) / len(team) * 0.5)
    return sim


def vote(reads, teams):
    """reads: list of (text, conf). Returns sorted [(team, score)] best first."""
    scores = defaultdict(float)
    for text, conf in reads:
        for team in teams:
            sim = similarity(text, team)
            if sim >= MIN_SIM:
                scores[team] += conf * sim
    return sorted(scores.items(), key=lambda kv: -kv[1])


# ----------------------------------- main ----------------------------------- #
def main():
    crop_dir, out_dir = Path(CROP_DIR), Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p
        for p in crop_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not paths:
        sys.exit(f"No images found in {crop_dir}")

    reader = easyocr.Reader(["en"], gpu=True)  # falls back to CPU if no GPU

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name}: unreadable")
            continue

        bumper, alliance, mask = find_bumper(img)
        cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), mask)
        if bumper is None or bumper.size == 0:
            print(f"{path.name}: no bumper found")
            continue
        cv2.imwrite(str(out_dir / f"{path.stem}_bumper.png"), bumper)

        reads = []
        for name, variant in preprocess_variants(bumper).items():
            cv2.imwrite(str(out_dir / f"{path.stem}_{name}.png"), variant)
            for _, text, conf in reader.readtext(
                variant, allowlist="0123456789", detail=1
            ):
                text = text.strip()
                if text:
                    reads.append((text, float(conf)))

        ranking = vote(reads, TEAMS)
        raw = ", ".join(f"{t}({c:.2f})" for t, c in reads) or "-"
        if ranking:
            best = ranking[0]
            runner = ranking[1][1] if len(ranking) > 1 else 0.0
            print(
                f"{path.name}: {alliance} | best={best[0]} score={best[1]:.2f} "
                f"margin={best[1] - runner:.2f} | raw: {raw}"
            )
        else:
            print(f"{path.name}: {alliance} | no team matched | raw: {raw}")


if __name__ == "__main__":
    main()
