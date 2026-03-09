# Fairness Analysis of Machine Learning Models using IBM AIF360

## Project Overview
This repository contains a Master's-level socio-technical evaluation and mitigation of algorithmic bias in the German Credit Dataset using the IBM AI Fairness 360 (AIF360) toolkit. The protected attribute analyzed is **Age** (Privileged: 25+, Unprivileged: Under 25). 

## Files in this Repository
* `Fairness_Assignment.ipynb`: The Jupyter Notebook containing EDA, model training, fairness evaluation (SPD, DI, EOD, AOD), and bias mitigation via Reweighing.
* `Fairness_Metrics_Report.pdf`: A comprehensive 7-page academic report detailing the socio-technical justification, methodology, results, and MLOps deployment considerations.

## Reproducibility Instructions
To run this code locally and reproduce the findings:
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook cells in sequential order. 

**Dependencies:**
* `aif360`
* `scikit-learn`
* `pandas`
* `numpy`
* `matplotlib`
* `seaborn`