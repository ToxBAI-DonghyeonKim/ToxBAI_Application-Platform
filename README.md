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

----------------------------------------------------------------------------------------------------------------------------------

## 입력 명세서 초안

파일명 예시는 `INPUT_SPECIFICATION.md`로 두면 좋습니다.

```markdown
# Prediction Input Specification

## Purpose
This document describes the input format required to run prediction in the ToxBAI Application Platform.

Users must prepare an Excel file based on:
- `prediction_input_template.xlsx`

## Input File Format
- File type: `.xlsx`
- Required template: `prediction_input_template.xlsx`

The input file must follow the same sheet structure and column format as the template.

## Input Contents
The prediction input file is used to provide:
1. compound information
2. assay information for prediction

## Compound Information
Users should provide the compounds to be evaluated.

Typical required fields are expected to include:
- compound identifier
- chemical structure representation such as SMILES

If multiple compounds are submitted, each compound should occupy one row.

## Assay Information
Users should specify which assay predictions they want to run.

Typical required fields are expected to include:
- assay name or assay list
- target model group if needed

Only assays supported by the models stored in this repository can be predicted.

## Required Rule
Users must not modify:
- sheet names
- required column names
- overall input template structure

Changing the template structure may cause the prediction script to fail.

## Expected Workflow
1. Copy `prediction_input_template.xlsx`
2. Enter compound information
3. Enter assay selection information
4. Save the Excel file
5. Run the prediction pipeline
6. Check the generated output file

## Important Constraints
- Input must follow the template exactly.
- Unsupported assay names may fail during prediction.
- Invalid or unparsable chemical structures may fail during fingerprint generation.
- The system assumes that the requested assays correspond to existing pre-trained models.

## Recommended Validation Before Run
Before execution, confirm:
- the Excel file is saved in `.xlsx` format
- all required sheets are present
- all mandatory columns are filled
- SMILES strings are valid
- assay names match supported model names

## Platform Integration Notes
For implementation in an open platform, the following UI/API fields are recommended:
- compound_id
- smiles
- assay_name

The platform should validate:
- empty fields
- invalid SMILES
- unsupported assay names
- duplicated records if necessary
