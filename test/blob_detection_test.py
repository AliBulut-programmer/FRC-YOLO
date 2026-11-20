"""import os
import sys

import cv2
import numpy as np

VIDEO_PATH = "C:/Users/alibu/Desktop/FRC-YOLO/unannotated_vids/Einstein Final 2 - FIRST Championship - FIRST Robotics Competition_nSzBX1nPtWE.mp4"
OUTPUT_PATH = os.path.join("outputs", "test_blob.mp4")


def yellow_candidates(
    frame, lo=(15, 120, 150), hi=(35, 255, 255), min_area=2, max_area=400, min_circ=0.2
):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (min_area <= a <= max_area):
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * a / (per * per + 1e-6)
        if circ >= min_circ:
            (x, y), _ = cv2.minEnclosingCircle(c)
            out.append((x, y, a))
    return out


def _draw_rois(frame, rois):
    overlay = frame.copy()
    for x, y, width, height in rois:
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width - 1, y + height - 1),
            (255, 255, 255),
            thickness=cv2.FILLED,
        )
    cv2.addWeighted(overlay, 0.06, frame, 0.94, 0, dst=frame)
    for x, y, width, height in rois:
        cv2.rectangle(
            frame,
            (x, y),
            (x + width - 1, y + height - 1),
            (180, 180, 180),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return frame


def _is_in_rois(x, y, rois):
    return any(
        roi_x <= x < roi_x + width and roi_y <= y < roi_y + height
        for roi_x, roi_y, width, height in rois
    )


def detect_video(
    video_path=VIDEO_PATH,
    output_path=OUTPUT_PATH,
    seed_frame_index=2000,
    stop_frame_index=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Could not determine a valid frame rate for: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Could not determine the frame count for: {video_path}")

    if not 0 <= seed_frame_index < frame_count:
        cap.release()
        raise ValueError(f"Seed frame must be between 0 and {frame_count - 1}")

    if stop_frame_index is None:
        stop_frame_index = int(
            input(f"Enter the earliest frame to analyze (0-{seed_frame_index - 1}): ")
        )
    if not 0 <= stop_frame_index < seed_frame_index:
        cap.release()
        raise ValueError(f"Stop frame must be between 0 and {seed_frame_index - 1}")

    segment_length = seed_frame_index - stop_frame_index + 1

    writer = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, seed_frame_index)
        ok, seed_frame = cap.read()
        if not ok:
            raise OSError(f"Could not read seed frame {seed_frame_index}")

        rois = []
        for roi_index in range(2):
            roi = cv2.selectROI(
                f"Select area {roi_index + 1} (SPACE/ENTER to confirm)",
                seed_frame,
                showCrosshair=True,
                fromCenter=False,
            )
            x, y, width, height = map(int, roi)
            cv2.destroyAllWindows()
            if width <= 0 or height <= 0:
                raise ValueError("Both seed areas must be selected")
            rois.append((x, y, width, height))

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        height, width = seed_frame.shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise OSError(f"Could not create output video: {output_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, stop_frame_index)
        for segment_index in range(segment_length):
            frame_index = stop_frame_index + segment_index
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"Could not read video frame {frame_index}")

            _draw_rois(frame, rois)
            detections = yellow_candidates(frame)
            roi_detections = 0
            for x, y, _ in detections:
                in_roi = _is_in_rois(x, y, rois)
                roi_detections += int(in_roi)
                color = (255, 0, 255) if in_roi else (0, 255, 0)
                cv2.circle(frame, (int(x), int(y)), 7, color, 2)

            cv2.putText(
                frame,
                f"Blobs: {len(detections)} | In selected areas: {roi_detections}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Frame: {frame_index}/{seed_frame_index}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            writer.write(frame)

            if segment_index % 100 == 0:
                print(f"Processing detections: {segment_index + 1}/{segment_length}")
        print(f"Saved processed video: {output_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    output_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH
    seed_frame_index = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    stop_frame_index = int(sys.argv[4]) if len(sys.argv) > 4 else None
    detect_video(
        video_path,
        output_path,
        seed_frame_index=seed_frame_index,
        stop_frame_index=stop_frame_index,
    )
"""

import cv2
import numpy as np
import os
import sys

VIDEO_PATH = "C:/Users/alibu/Desktop/FRC-YOLO/unannotated_vids/Einstein Final 2 - FIRST Championship - FIRST Robotics Competition_nSzBX1nPtWE.mp4"
OUTPUT_PATH = os.path.join("outputs", "test_blob.mp4")
START_FRAME = 300
END_FRAME = 2000
# Lower these if touching fuel remains merged; raise them if one fuel splits into noise.
MIN_PEAK_DISTANCE = 7
PEAK_THRESHOLD_RATIO = 0.35
LOWER_HSV = (20, 120, 150)
UPPER_HSV = (35, 255, 255)
MORPH_KERNEL_SIZE = 3
MIN_CONTOUR_AREA = 6
MAX_CONTOUR_AREA = 400
MIN_CIRCULARITY = 0.5


def _component_contours(frame, component_mask):
    padded_mask = cv2.copyMakeBorder(
        component_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0
    )
    distance = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
    peak_floor = distance.max() * PEAK_THRESHOLD_RATIO
    dilated_distance = cv2.dilate(distance, np.ones((3, 3), np.uint8))
    local_maxima = (distance >= dilated_distance - 1e-6) & (distance >= peak_floor)
    local_maxima &= padded_mask > 0

    peak_count, peak_labels = cv2.connectedComponents(local_maxima.astype(np.uint8))
    peaks = []
    for peak_label in range(1, peak_count):
        ys, xs = np.where(peak_labels == peak_label)
        if len(xs) == 0:
            continue
        strongest = np.argmax(distance[ys, xs])
        peaks.append(
            (
                int(xs[strongest]),
                int(ys[strongest]),
                float(distance[ys[strongest], xs[strongest]]),
                peak_labels == peak_label,
            )
        )

    peaks.sort(key=lambda peak: peak[2], reverse=True)
    separated_peaks = []
    for peak in peaks:
        if all(
            np.hypot(peak[0] - chosen[0], peak[1] - chosen[1]) >= MIN_PEAK_DISTANCE
            for chosen in separated_peaks
        ):
            separated_peaks.append(peak)

    if len(separated_peaks) < 2:
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    markers = np.ones(padded_mask.shape, dtype=np.int32)
    markers[padded_mask > 0] = 0
    for marker_id, peak in enumerate(separated_peaks, start=2):
        markers[peak[3]] = marker_id

    padded_frame = cv2.copyMakeBorder(
        frame,
        1,
        1,
        1,
        1,
        cv2.BORDER_REPLICATE,
    )
    watershed_labels = cv2.watershed(padded_frame, markers)
    split_contours = []
    for marker_id in range(2, len(separated_peaks) + 2):
        segment = np.uint8((watershed_labels == marker_id) & (padded_mask > 0)) * 255
        segment = segment[1:-1, 1:-1]
        contours, _ = cv2.findContours(
            segment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        split_contours.extend(contours)

    if not split_contours:
        split_contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    return split_contours


def yellow_candidates(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    candidates = []
    for component_id in range(1, component_count):
        x, y, width, height, _ = stats[component_id]
        component_mask = (
            np.uint8(labels[y : y + height, x : x + width] == component_id) * 255
        )
        component_frame = frame[y : y + height, x : x + width]
        contours = _component_contours(component_frame, component_mask)

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
            if circularity < MIN_CIRCULARITY:
                continue
            (center_x, center_y), _ = cv2.minEnclosingCircle(contour)
            candidates.append((center_x + x, center_y + y, area))
    return candidates


def run_detection_video(video_path=VIDEO_PATH, output_path=OUTPUT_PATH):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    writer = None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(
                f"Could not determine a valid frame rate for: {video_path}"
            )
        if not 0 <= START_FRAME <= END_FRAME < frame_count:
            raise ValueError(
                f"Requested frames {START_FRAME}-{END_FRAME}; video has {frame_count} frames"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
        ok, frame = cap.read()
        if not ok:
            raise OSError(f"Could not read video frame {START_FRAME}")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise OSError(f"Could not create output video: {output_path}")

        for frame_index in range(START_FRAME, END_FRAME + 1):
            if frame_index > START_FRAME:
                ok, frame = cap.read()
                if not ok:
                    raise OSError(f"Could not read video frame {frame_index}")

            detections = yellow_candidates(frame)
            for x, y, area in detections:
                radius = max(3, int(np.sqrt(area / np.pi)))
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    radius,
                    (0, 255, 0),
                    2,
                )

            cv2.putText(
                frame,
                f"Frame {frame_index} | Blobs detected: {len(detections)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(frame)

            processed = frame_index - START_FRAME + 1
            if processed % 100 == 0 or frame_index == END_FRAME:
                print(f"Processing frames: {processed}/{END_FRAME - START_FRAME + 1}")

        print(f"Saved detection video: {output_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    input_video = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    output_video = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PATH
    run_detection_video(input_video, output_video)
