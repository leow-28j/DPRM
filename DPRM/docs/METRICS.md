# Metric definitions

`utils/scd_metrics.py` reports two groups of values:

- Date-wise diagnostics (`date_1`, `date_2`, `mean_overall_accuracy`, and
  `mean_iou`) evaluate the two semantic maps separately.
- `paper_metrics` implements the state-based semantic change detection (SCD)
  protocol used for DPRM reporting. These are the values to use when comparing
  OA, Fscd, mIoU, and SeK in the paper.

## State construction

For `C` semantic classes, a pixel with unchanged semantic labels is assigned
state `0`. A changed pixel with ordered labels `(T1, T2)` is assigned one of
the `C × C` transition states. Predictions and references are then accumulated
into one transition-state confusion matrix. Invalid labels and `ignore_index`
pixels are excluded.

## Reported paper metrics

| Name | Definition |
| --- | --- |
| OA | Exact transition-state accuracy over all valid pixels. |
| Fscd | Harmonic mean of transition-state precision and recall. A changed prediction counts as correct only when its full ordered semantic transition is correct. |
| mIoU | Mean IoU of the collapsed unchanged and changed groups. Within the changed group, any predicted changed transition overlaps any reference changed transition. |
| SeK | `Kappa_n0 × exp(IoU_changed) / e`, where `Kappa_n0` is Cohen's kappa after removing unchanged true negatives from the transition-state confusion matrix. |

The output also includes `Kappa_n0` and `binary_iou_changed` to make SeK
auditable. Values are proportions in `[0, 1]`, not percentages.

The utility derives change/no-change from whether the two supplied semantic
maps differ. If an experiment uses an independent binary change head, convert
its outputs to the corresponding transition-state maps before applying this
utility.
