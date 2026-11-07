# Bank Marketing Subscription Prediction

This project explores whether a bank customer will subscribe to a term deposit following a marketing campaign. It uses exploratory data analysis and machine-learning classification models, including neural networks, to study the prediction task.

## Project contents

- [Final analysis notebook](Neural_Network.ipynb) — data preparation, analysis, model training, and evaluation.
- [Bank Marketing dataset](bank-full.csv) — dataset used by the analysis.

## Analysis overview

The notebook covers:

- Data inspection and consistency checks
- Exploratory analysis of customer and campaign attributes
- Outlier treatment, feature engineering, and categorical encoding
- Feature scaling and a stratified train/test split
- Logistic Regression and feedforward neural-network models
- Hyperparameter tuning, stratified cross-validation, and permutation feature importance
- A TensorFlow/Keras model with dropout, batch normalization, and early stopping

The target variable is whether a customer subscribed (`yes` or `no`). The target is imbalanced, so the notebook evaluates models with classification metrics and ROC-AUC rather than relying on accuracy alone.

## Reported results

The notebook reports:

| Model or evaluation | ROC-AUC |
|---|---:|
| Logistic Regression baseline | 0.891 |
| Scikit-learn MLP | 0.896 |
| 5-fold stratified cross-validation (MLP) | 0.8991 ± 0.0032 |
| TensorFlow/Keras model | 0.9002 |

These are the results recorded in the notebook; results may vary with changes to the data or execution environment.

## Requirements

- Python 3
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow

Install the packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
