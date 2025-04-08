# Integrating Machine Learning Phenology into CLM

This repository contains code and resources for integrating a machine learning-based phenology model into the Community Land Model (CLM). This work bridges data-driven modeling with process-based earth system modeling to improve phenological predictions.

## Overview

Phenology plays a critical role in terrestrial carbon, water, and energy exchanges. Traditional phenology schemes in land surface models are often limited in flexibility and accuracy. Here, we develop a PyTorch-based phenology model and interface it with CLM's phenology routines written in Fortran.

## Repository Structure

- `notebooks/`: Jupyter notebooks for exploratory analysis and model development.
- `src/`: Core Python code for data preprocessing, training, and inference.
- `fortran/`: Fortran code interfacing the ML phenology model with CLM.
- `scripts/`: Shell scripts for running model workflows or tests.
- `models/`: Trained model checkpoints and weight files.
- `data/`: Placeholder for datasets (not tracked in Git).
- `docs/`: Project documentation, notes, and integration guides.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/phenology-ml-clm.git
   cd phenology-ml-clm
   ```

2. (Optional) Create a Python environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## CLM Integration

Coming soon.

## Contributing

Please open an issue or submit a pull request for improvements or suggestions.

## License

This project is released under [Your License Here].

## Acknowledgments

This project is supported by [Your Institution / NCAR / NSF or other orgs].  
Model development leverages [PyTorch](https://pytorch.org/) and builds upon the CLM framework from [CESM](https://www.cesm.ucar.edu/).
