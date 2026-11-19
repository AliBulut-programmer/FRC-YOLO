import numpy as np


class TrackletStitcher:
    def __init__(self, n_robots=6):
        if n_robots < 1:
            raise ValueError("n_robots must be at least 1")
        self.n_robots = n_robots  # number of actual robots in the scene

    def build_galleries(self, tracker_ids, embeddings):
        """Creates galleries, which associates tracked_ids to mean embedding vectors

        Args:
            tracker_ids (_type_): _description_
            embeddings (_type_): _description_

        Returns:
            _dict_: returns a galleries  dict that holds representative emb values of tracklets where
        keys are the unique tracker ids \n
        values are the mean embedding vectors
        """
        tracker_ids = np.asarray(tracker_ids)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if tracker_ids.ndim != 1 or embeddings.ndim != 2:
            raise ValueError("tracker_ids must be 1D and embeddings must be 2D")
        if len(tracker_ids) != len(embeddings):
            raise ValueError("tracker_ids and embeddings must have the same length")

        # galleries is a dict that holds representative emb values of tracklets where
        #! keys are the unique tracker ids
        #! values are the mean embedding vectors
        galleries = {}
        for tid in np.unique(tracker_ids):
            if tid < 0:  # -1 ids dont have embeddings assigned
                continue
            # create boolean mask, filters for current tid in the .npz file order
            mask = tracker_ids == tid
            # get embeddings of current tid
            tid_embs = embeddings[mask]
            valid = np.isfinite(tid_embs).all(axis=1)  # check emb validity
            if valid.any():  # check if there are any embeddings at all
                galleries[int(tid)] = tid_embs[valid].mean(axis=0)  # create galleries
        return galleries

    def stitch(
        self, galleries, cooccurring_pairs=(), tracklet_support=None, max_dist=0.2
    ):
        """Merge similar non-overlapping tracklets, leaving excess tracks unmapped."""
        tids = sorted(galleries)
        if not tids:
            return {}
        if not 0 <= max_dist <= 2:
            raise ValueError("max_dist must be between 0 and 2")

        matrix = np.asarray([galleries[tid] for tid in tids], dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Each gallery must be a 1D embedding vector")
        if not np.isfinite(matrix).all():
            raise ValueError("Gallery embeddings must contain only finite values")

        norms = np.linalg.norm(matrix, axis=1)
        normalized = np.zeros_like(matrix)
        nonzero = norms > 0
        normalized[nonzero] = matrix[nonzero] / norms[nonzero, None]
        distances = np.clip(1.0 - normalized @ normalized.T, 0.0, 2.0)
        distances[~nonzero, :] = 1.0
        distances[:, ~nonzero] = 1.0

        cannot_link = {
            (min(int(first), int(second)), max(int(first), int(second)))
            for first, second in cooccurring_pairs
            if first != second
        }
        support = tracklet_support or {}
        clusters = [{tid} for tid in tids]
        positions = {tid: index for index, tid in enumerate(tids)}

        while True:
            best_merge = None
            for first_index in range(len(clusters)):
                for second_index in range(first_index + 1, len(clusters)):
                    first_cluster = clusters[first_index]
                    second_cluster = clusters[second_index]
                    if any(
                        (min(first, second), max(first, second)) in cannot_link
                        for first in first_cluster
                        for second in second_cluster
                    ):
                        continue

                    distance = float(
                        np.mean(
                            [
                                distances[positions[first], positions[second]]
                                for first in first_cluster
                                for second in second_cluster
                            ]
                        )
                    )
                    candidate = (
                        distance,
                        min(first_cluster),
                        first_index,
                        second_index,
                    )
                    if distance <= max_dist and (
                        best_merge is None or candidate < best_merge
                    ):
                        best_merge = candidate

            if best_merge is None:
                break

            _, _, first_index, second_index = best_merge
            clusters[first_index].update(clusters[second_index])
            del clusters[second_index]

        if len(clusters) > self.n_robots:
            clusters.sort(
                key=lambda cluster: (
                    -sum(support.get(tid, 1) for tid in cluster),
                    min(cluster),
                )
            )
            clusters = clusters[: self.n_robots]

        clusters.sort(key=min)
        return {
            tid: cluster_id
            for cluster_id, cluster in enumerate(clusters)
            for tid in cluster
        }
