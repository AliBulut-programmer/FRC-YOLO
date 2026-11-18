import unittest

import numpy as np

from src.track_detect.robotposition.global_id_resolution.stitcher import (
    TrackletStitcher,
)


class TrackletStitcherTests(unittest.TestCase):
    def test_build_galleries_ignores_untracked_sentinel(self):
        tracker_ids = np.array([-1, 0, 0])
        embeddings = np.array([[0, 1], [1, 0], [3, 0]], dtype=np.float32)

        galleries = TrackletStitcher().build_galleries(tracker_ids, embeddings)

        self.assertNotIn(-1, galleries)
        np.testing.assert_array_equal(galleries[0], [2, 0])

    def test_merges_similar_nonoverlapping_tracks_only(self):
        galleries = {
            0: np.array([1, 0], dtype=np.float32),
            1: np.array([0, 1], dtype=np.float32),
            2: np.array([0.99, 0.01], dtype=np.float32),
        }

        mapping = TrackletStitcher(n_robots=2).stitch(
            galleries, cooccurring_pairs={(0, 1)}
        )

        self.assertEqual(mapping[0], mapping[2])
        self.assertNotEqual(mapping[0], mapping[1])

    def test_leaves_low_support_conflicting_track_unmatched(self):
        galleries = {
            0: np.array([1, 0], dtype=np.float32),
            1: np.array([0, 1], dtype=np.float32),
            2: np.array([-1, 0], dtype=np.float32),
        }

        mapping = TrackletStitcher(n_robots=2).stitch(
            galleries,
            cooccurring_pairs={(0, 1), (0, 2), (1, 2)},
            tracklet_support={0: 10, 1: 8, 2: 1},
        )

        self.assertEqual(set(mapping), {0, 1})
        self.assertEqual(set(mapping.values()), {0, 1})


if __name__ == "__main__":
    unittest.main()
