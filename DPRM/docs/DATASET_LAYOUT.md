# Semantic change detection dataset layout

The validator accepts a dataset root with one or more split directories (for
example, `train`, `val`, or `test`). Each checked split must contain paired
images and semantic labels using this layout:

```text
DATASET_ROOT/
  train/
    T1/       # first-date images
    T2/       # second-date images
    GT_T1/    # first-date integer class-ID masks
    GT_T2/    # second-date integer class-ID masks
    GT_CD/    # optional binary change masks
  val/
    ...
  test/
    ...
```

Files are matched by relative filename. Class masks should be single-channel
integer images. `GT_CD`, when present, should be a binary mask; it is not
required by the metric utility because binary change can be derived from the
two semantic masks.

Datasets are not bundled with this repository. Download them only from their
official providers and follow their license, access, and citation terms.
