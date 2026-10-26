# First-stage public release notes

## Included

- `utils/scd_metrics.py`: standalone NumPy implementation of binary-change
  and two-date semantic metrics.
- `tools/validate_dataset_layout.py`: checks a local SCD dataset directory
  without reading or copying its images.
- `docs/DATASET_LAYOUT.md`: a neutral directory convention for the validator.
- `docs/METRICS.md` and `tests/test_scd_metrics.py`: metric definitions and
  regression tests for the SCD reporting protocol.

## Intentionally excluded

- all DPRM modules and their six model integrations;
- training, inference, validation, profiling, and ablation scripts tied to
  DPRM;
- model weights, datasets, predictions, logs, and experiment records; and
- copied code from baseline projects pending provenance and attribution audit.

## Pre-publication checklist

1. Merge the text in `README.md` into the existing GitHub README rather than
   overwriting its existing content.
2. Choose and add a repository license after confirming institutional and
   contributor ownership.
3. Re-audit every proposed figure and third-party dependency before adding it.
4. After acceptance, add the implementation and reproduce instructions in a
   separate reviewed release.
