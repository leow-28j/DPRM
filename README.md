
# Difference-Preserving Residual Modulation (DPRM)

Official repository for the paper:

Difference-Preserving Residual Modulation: A Plug-and-Play Bi-Temporal Interaction Module for Semantic Change Detection

## Overview

DPRM is a lightweight, plug-and-play bi-temporal interaction module designed for remote sensing semantic change detection (SCD).

The proposed module combines explicit difference extraction, context-guided difference modulation, and shared residual updates to enhance bi-temporal feature representations while preserving their original element-wise signed differences at the module output.

DPRM can be integrated into CNN-, Transformer-, and Mamba-based SCD architectures without modifying their original decoders or training objectives.

## Experimental Evaluation

DPRM has been evaluated on two semantic change detection datasets:

- SECOND
- Hi-UCD mini

Experiments were conducted using six representative SCD baseline models:

- SCDNet
- BiSRNet
- ChangeMask
- SCanNet
- ChangeMamba
- FoBa-Mamba

## Code Availability

This repository currently provides preliminary project resources and selected non-core components.

The complete DPRM implementation, backbone integration code, and relevant experimental configurations are planned for release following acceptance of the manuscript.

## Citation

Citation information will be updated upon publication.
