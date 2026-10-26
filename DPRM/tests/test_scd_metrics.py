"""Regression tests for the standalone SCD metric definitions."""

from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from scd_metrics import scd_paper_metrics  # noqa: E402


class SCDPaperMetricsTest(unittest.TestCase):
    def test_perfect_prediction_is_one(self) -> None:
        target_t1 = np.array([[0, 0, 1, 2], [1, 2, 0, 2]])
        target_t2 = np.array([[0, 2, 1, 0], [2, 2, 1, 0]])
        result = scd_paper_metrics(target_t1, target_t2, target_t1, target_t2, 3)
        for name in ("OA", "Fscd", "mIoU", "SeK", "Kappa_n0", "binary_iou_changed"):
            self.assertAlmostEqual(result[name], 1.0, places=12)

    def test_wrong_changed_transition_reduces_fscd_not_change_overlap(self) -> None:
        target_t1 = np.array([[0, 0], [1, 2]])
        target_t2 = np.array([[0, 1], [2, 2]])
        pred_t1 = np.array([[0, 0], [1, 2]])
        pred_t2 = np.array([[0, 2], [2, 2]])
        result = scd_paper_metrics(pred_t1, pred_t2, target_t1, target_t2, 3)
        self.assertAlmostEqual(result["OA"], 0.75, places=12)
        self.assertAlmostEqual(result["Fscd"], 0.5, places=12)
        self.assertAlmostEqual(result["binary_iou_changed"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
