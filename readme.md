# ❤️ Heart Disease Prediction using Stacked Ensembled Model with Modified Whale Optimization Algorithm

This project predicts the likelihood of heart disease using a **stacked ensemble** of machine learning models, with a **Modified Whale Optimization Algorithm (MWOA)** for feature selection and/or hyperparameter tuning. The workflow covers data preprocessing, scaling, model training, optimization, and evaluation.

---

## 🔍 What’s Inside
- **Modified Whale Optimization Algorithm (MWOA):**
  - Used to search the space of feature subsets and/or model hyperparameters.
  - Fitness function based on validation performance (e.g., accuracy/AUC).
  - Supports constraints (e.g., minimum features) and early stopping.
- **Stacked Ensemble Classifier:**
  - Base learners (examples): SVM, Random Forest, Extra Trees, Gradient Boosting/AdaBoost.
  - **Meta-learner**: typically Logistic Regression over out-of-fold predictions.
- **Robust Preprocessing:**
  - Missing value handling (if present).
  - **MinMaxScaler** normalization for model stability.
- **Evaluation & Visualization:**
  - Accuracy, Confusion Matrix, ROC Curve, AUC.
  - Cross-validation for reliable estimates.

---

## 🧮 Method Overview

### 1. Preprocess
- Load `data/heart.csv`
- Handle missing values (if any)
- Split into train/test sets
- Scale features with **MinMaxScaler**

### 2. Define Models
- **Base learners:**
  - Support Vector Classifier (SVC)
  - RandomForestClassifier
  - ExtraTreesClassifier
  - GradientBoostingClassifier / AdaBoostClassifier
- **Meta-learner:**
  - LogisticRegression trained on out-of-fold (OOF) predictions

### 3. Optimize with Modified Whale Optimization Algorithm (MWOA) *(Optional)*
- **Search Space:**
  - Feature mask (binary vector) → if feature selection enabled
  - Hyperparameters for base/stacked models → if tuning enabled
- **Fitness Function:**
  - Validation AUC/Accuracy via cross-validation
- **Output:**
  - Best feature subset  
  - Best hyperparameters  

### 4. Train Stacked Ensemble
- Fit base learners on the full training set
- Generate **OOF predictions**
- Fit meta-learner on OOF predictions

### 5. Evaluate
- Accuracy Score  
- Confusion Matrix  
- ROC Curve & AUC on the test set  
- *(Optional)* Calibration plots / Feature importance (if supported)
