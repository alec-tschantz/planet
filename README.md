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
- Test with fully observed
- Do metric plotting
- Do model / data loading
- Move validate & test to same function to save compute
- Move all environments to same class -> deal with necessary differences
- Use visdoom [https://github.com/shakenes/vizdoomgym]
- Use [pybullet](https://github.com/benelot/pybullet-gym)
- Normalization
- _Error_: Buffer does not work if buffer is less than sequence length