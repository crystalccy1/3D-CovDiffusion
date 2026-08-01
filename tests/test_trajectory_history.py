import unittest

import numpy as np

from covdiffusion.common.trajectory_history import previous_action_index


class PreviousActionIndexTest(unittest.TestCase):
    def test_first_sample_never_uses_previous_episode(self):
        valid = np.array([True, True, True, True, True])
        self.assertEqual(previous_action_index(3, 3, valid), 3)

    def test_uses_latest_valid_action_inside_current_episode(self):
        valid = np.array([True, True, True, True, False, True])
        self.assertEqual(previous_action_index(6, 3, valid), 5)

    def test_does_not_cross_boundary_when_local_history_is_padding(self):
        valid = np.array([True, True, True, False, False, True])
        self.assertEqual(previous_action_index(5, 3, valid), 5)


if __name__ == "__main__":
    unittest.main()
