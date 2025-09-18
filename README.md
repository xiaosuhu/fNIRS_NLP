# fNIRS_NLP — fNIRS pipelines for natural language processing (NLP)

A collection of MATLAB and Python scripts to preprocess, analyze, and run NLP-based analyses on fNIRS data. The project includes data-preparation tools, surprisal integration, GPT-2 embedding forecasting, and example results.

<p align="left">
  <img src="https://img.shields.io/badge/version-0.1-blue" alt="Version Badge">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License Badge">
  <img src="https://img.shields.io/badge/build-building_inprogress-yellow" alt="Build Badge">
</p>

Short summary
- Example (de-identified) data and processed cutted data used for testing and demos
- Scripts for data preparation and three-way suprisal calculations
- NLP forecasting pipeline (GPT-2 embeddings, design matrix prep, brain/forecast scores)
- Example results and figures (see `Example_data/`)

---

## Quick start

Requirements
- MATLAB (with the **NIRS toolbox** and **GLM pipeline** available on the MATLAB path)
- Python 3.10+ (recommended) and the Python packages listed in `NLP_Forcasting/requirements.txt`

1) MATLAB: add the required toolboxes and repo paths to MATLAB's path (see Requirements)
2) Python (Windows PowerShell example):

```powershell
conda create -n fnirs_nlp python=3.10 -y ; conda activate fnirs_nlp
conda install --yes --file NLP_Forcasting/requirements.txt
```

Notes
- On Windows run the Miniconda installer executable from https://docs.conda.io/en/latest/miniconda.html.
- The `requirements.txt` in `NLP_Forcasting/` was exported from a conda environment — restoring it via `conda install --file` recreates pinned versions.

### Installing Miniconda

To set up the Python environment, you need Miniconda installed. Follow these steps:

1. Download the Miniconda installer for your operating system from the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html).
2. Run the installer and follow the on-screen instructions to complete the installation.
3. After installation, open a terminal (or Anaconda Prompt on Windows) and verify the installation by running:

   ```bash
   conda --version
   ```

   This should display the installed version of Conda.

Once Miniconda is installed, proceed with the steps below to create and activate the required environment.

---

## What this repository contains

- `Data_Prep/` — MATLAB scripts for cutting and preparing raw fNIRS files (e.g. `data_cut_two_ver_combined.m`)
- `Example_data/` — example dataset, processed files, design matrices, and example results for demos
- `NLP_Forcasting/` — Python notebooks and MATLAB scripts for embeddings extraction, design-matrix prep, brain score calculations, and plotting (see `brain_score.py` and notebooks)
- `Three_way_split/` — suprisal-based pipelines and GLM analysis scripts

---

## Suprisal / LingPred note
This repo references the LingPred toolkit (https://github.com/cnllab/lingpred) for surprisal extraction. Access to LingPred is managed by Dr. Jonathan Brennan (University of Michigan); please contact him for access and follow LingPred's setup instructions before running surprisal-related steps.

---

## Usage examples

1) MATLAB data-prep (example)

- Open MATLAB, add this repo and the required toolboxes to the MATLAB path, then run:

```matlab
% example (MATLAB)
Data_Prep/data_cut_two_ver_combined.m

% This code included cutting out the Q&A part (4 and 12 questions version) for the story fNIRS file
```

2) 3-Way split analysis

- Open `Three_way_split\story_surprisal_three_way_calc` and follow the notebook `run_on_test_sentence.ipynb` to extract 3-level surprisal using `ling-pred` toolbox.

- GLM analysis `Three_way_split\GLM_analysis`
   1. Use the recoding to add the surprisals into the fNIRS data files
   2. The first, group level analysis and plot are the same as in the GLM pipeline

3) Forcasting analysis

- Analyse using the following steps 0 - 6:
   1. Step 0 time matrix round
   2. GPT2 embedding calculation
   3. HbO data prep
   4. Design matrix prep
   5. brain score calc  
        Nested Cross-Validation and Ridge Regression
         - **Data Preparation**:
           - The input data (X, y) is scaled using `StandardScaler` to standardize features.
           - A `TimeSeriesSplit` is used for time-aware cross-validation.
         - **Nested Cross-Validation**:
           - For each fold in the outer cross-validation:
             - The training set is used to find the best ridge regression regularization parameter (`alpha`) via an inner cross-validation (`nested_tscv_ridge_cuda`).
             - In the inner loop, ridge regression is performed for multiple `alpha` values, and the one minimizing the mean squared error (MSE) across folds is selected.
         - **Ridge Regression**:
           - Using the best `alpha`, ridge regression is solved analytically on the GPU for the training data.
           - Predictions are generated for the test set.
         - **Correlation and Fisher Z Transformation**:
           - The Pearson correlation between the true and predicted values for the test set is computed using `compute_correlation`.
           - The correlation is transformed into a Fisher Z-score using `fisher_z_transform`.
   6. brain score plot
   7. ROI plot in Matlab
   8. Peak map plot in Matlab

   ![Forecasting Pipeline](Forcasting_pipeline.png)

Example results are included in `Example_data/`.

---

## Reproducibility & data

- Example de-identified data is included under `Example_data/` for demonstration only. If you use your own data, follow the same file naming and channel conventions present in the examples.
- Some analysis steps require external access (e.g., LingPred for surprisal). Obtain necessary permissions before running those steps.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Contact

For questions or collaborations contact:

- Frank Hu — xiaosuhu@umich.edu


