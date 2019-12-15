# PlaNet: Learning Latent Dynamics for Planning from Pixels
![status](https://img.shields.io/badge/status-development-orange)

This repository provides a reimplementation of the [Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet) repository, which is itself a reimplementation of [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) and the associated [code](https://github.com/google-research/planet). It was implemented for the purpose of understanding the model, and currently implements nothing over and above the previous implementations.

## Requirements
- `numpy`
- `torch`
- `gym`
- `matplotlib`

## Acknowledgements
- [@Kaixhin](https://github.com/Kaixhin) for [Kaixhin/PlaNet](https://github.com/Kaixhin/PlaNet), the basis for this implementation
- [@danijar](https://github.com/danijar) for [google-research/planet](https://github.com/google-research/planet)

## Tasks
- __Improvements__
  - Finesse model loading + checkpoint (no need for seed data)
  - Metrics - including time (option to save to GDrive)
  - Sort data set loading + big solution needed
  - Test with fully observed
  - Ensembles
  - Normalization

- __Environments__
  - Use visdoom [https://github.com/shakenes/vizdoomgym]
  - Use [pybullet](https://github.com/benelot/pybullet-gym)

- __Errors__
  - Buffer does not work if buffer is less than sequence length