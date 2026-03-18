# ToxBAI Application Platform

## Overview
This repository provides an inference-only platform for toxicity prediction using pre-trained models.

The repository is intended for platform integration and deployment, not for model training.  
Users prepare an input Excel file based on the provided template, run the prediction pipeline, and obtain assay-specific prediction results.

## Repository Structure
- `ToxCast models/`  
  Pre-trained ToxCast best-performance models
- `OECD TG models/`  
  Pre-trained OECD TG best-performance models
- `Predict_data.py`  
  Main prediction script
- `config.py`  
  Configuration file for model path, input/output path, and prediction settings
- `smiles2fing.py`  
  Molecular fingerprint generation module
- `common.py`  
  Common utility functions
- `prediction_input_template.xlsx`  
  Standard input template for prediction
- `launcher_local.sh`  
  Local execution script
- `requirements.txt`  
  Python dependency list

## Purpose
This repository is designed to support the following workflow:
1. The user prepares an Excel input file using `prediction_input_template.xlsx`.
2. The user enters compounds and assay targets in the template.
3. The prediction code generates molecular fingerprints from the input compounds.
4. The system loads the corresponding pre-trained best model for each requested assay.
5. The system returns prediction results in an output file.

## Input
Prediction requires an Excel file based on:
- `prediction_input_template.xlsx`

The input file should contain:
- compound information
- assay selection information

The exact sheet names and required columns must follow the template format.

## Output
The prediction pipeline produces assay-level prediction results for the input compounds.

Typical outputs may include:
- input compound identifier
- assay name
- predicted class or score
- model metadata if applicable

## Execution
### Option 1. Direct execution
Run the main prediction script after setting the required paths and options in `config.py`.

### Option 2. Local launcher
Run:
```bash
bash launcher_local.sh
