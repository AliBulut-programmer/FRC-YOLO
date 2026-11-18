import numpy as np
from scipy.spatial import distance
import supervision as sv
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


class MasterProcessor:
    def __init__(self, velocity_thresh, realtime, fps, init_frame_count):
        self.velocity_thresh = velocity_thresh
        self.raw_detections = {}
        self.processed_detections = {}
        self.realtime = realtime
        self.fps = fps
        self.velocity_history = {}
        self.init_frame_count = init_frame_count

        self.last_seen = {}
        self.last_frame = {}

    def update_raw_detections(self, detections, frame_count):
        self.raw_detections[frame_count] = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy(),
            class_id=detections.class_id.copy(),
            tracker_id=detections.tracker_id.copy(),
        )

    def update_processed_detections(self, detections, frame_count):
        print("updating processed detections")
        self.processed_detections[frame_count] = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy(),
            class_id=detections.class_id.copy(),
            tracker_id=detections.tracker_id.copy(),
        )

    def velocity_map(self, velocity_matrix, frame_count):
        self.velocity_history[frame_count] = velocity_matrix

    def velocity_calc(self, detections, frame_count, homography_matrix=None):
        # skip homography matrix for now
        velocity_matrix = {}
        center_positions = {}
        past_center_positions = {}
        # print(f"detections {detections}")
        if len(self.processed_detections) == 0:
            print("returning none")
            return None, None, None

        for idx in detections.tracker_id:
            if idx == -1:
                continue
            for i in range(frame_count - self.init_frame_count):
                past_detection = self.processed_detections.get(frame_count - (i + 1))
                # print(f"raw detections {self.raw_detections}")
                # print(f"processed detections {self.processed_detections}")
                # print(f"past detection {past_detection}")
                if past_detection == None:
                    continue
                masked_detection = past_detection[past_detection.tracker_id == idx]
                # print(f"masked detection {masked_detection}")
                if len(masked_detection) > 0:

                    # print("found masked detection")
                    # print(f"current detection {detections.xyxy[detections.tracker_id==idx][0]}")
                    x1, y1, x2, y2 = masked_detection.xyxy[0]
                    mxc = (x2 + x1) / 2
                    myc = (y2 + y1) / 2
                    past_center_positions[idx] = (int(mxc), int(myc))
                    x1, y1, x2, y2 = detections.xyxy[detections.tracker_id == idx][0]
                    cxc = (x2 + x1) / 2
                    cyc = (y2 + y1) / 2
                    center_positions[idx] = (int(cxc), int(cyc))
                    # print(f"{(mxc,myc)} ,,, {(cxc, cyc)}")
                    # print(f"distance {distance.euclidean((mxc,myc), (cxc, cyc))}")

                    dist = distance.euclidean((mxc, myc), (cxc, cyc))
                    time = i + 1  # number of frames
                    velocity = dist / time
                    velocity_matrix[idx] = velocity
                    break

        return velocity_matrix, center_positions, past_center_positions

    def velocity_filter(self, velocity_matrix, detections):
        keep = np.ones(len(detections), dtype=bool)

        for i, tid in enumerate(detections.tracker_id):
            velocity = velocity_matrix.get(tid)
            if tid != -1 and velocity is not None and velocity >= self.velocity_thresh:
                keep[i] = False
                print("velocity filter - fail")

        print("velocity filter worked")
        return detections[keep]

    def reassign_unassigned(self, detections):
        # check for empty detections
        if len(detections) == 0 or detections.tracker_id is None:
            return detections
        ids = detections.tracker_id  # ids in the current frame
        unassigned_idx = np.where((ids == -1) | (ids > 5))[
            0
        ]  # ids with -1 or >5 in the current frame
        present = set(ids.tolist())  # SET of all the current ids
        unused_canon = [c for c in range(6) if c not in present and c in self.last_seen]
        if (
            len(unassigned_idx) == 0 or len(unused_canon) == 0
        ):  # if all current indices are properly assigned* or there are no canon ids left to use
            return detections
        rows = detections.xyxy[unassigned_idx][
            :, :2
        ]  # get detections with unassigned indices and slice them so only the top left coords show up
        cols = np.vstack([self.last_seen[c] for c in unused_canon])
        cost_matrix = distance.cdist(rows, cols)
        r, c = linear_sum_assignment(cost_matrix)
        for ri, ci in zip(r, c):
            if cost_matrix[ri, ci] <= self.velocity_thresh:
                detections.tracker_id[unassigned_idx[ri]] = unused_canon[ci]
        return detections

    def drop_non_canonical(self, detections):
        if len(detections) == 0 or detections.tracker_id is None:
            return detections
        ids = detections.tracker_id
        return detections[(ids >= 0) & (ids <= 5)]

    def update_last_seen(self, detections):
        if len(detections) == 0 or detections.tracker_id is None:
            return detections

        for (
            tid,
            box,
        ) in zip(detections.tracker_id, detections.xyxy):
            self.last_seen[int(tid)] = box[:2].copy()

    def process_frame(self, frame, frame_count, detections):

        self.update_raw_detections(detections=detections, frame_count=frame_count)

        # velocity_matrix,center_positions,past_center_positions= self.velocity_calc(detections=detections, frame_count=frame_count)
        velocity_matrix, center_positions, past_center_positions = [
            None for i in range(3)
        ]

        if velocity_matrix is not None:
            if len(velocity_matrix) > 0:
                self.velocity_map(
                    velocity_matrix=velocity_matrix, frame_count=frame_count
                )
                detections = self.velocity_filter(velocity_matrix, detections)

        detections = self.reassign_unassigned(detections)
        # detections = self.drop_non_canonical(detections)   # removes leftover -1 / >5
        self.update_last_seen(detections)
        self.update_processed_detections(detections=detections, frame_count=frame_count)
        print(frame_count)
        return detections, velocity_matrix, center_positions, past_center_positions
