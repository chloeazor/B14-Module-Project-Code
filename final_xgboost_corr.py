# %% Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/files/Chloe')

# Custom helper functions (if needed; comment out if not used)
import help_functions as hf
hf.install_packages()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %% Step 0: Data preprocessing - Load Methylation data, descriptors, and demographics

# Load Methylation data
file_Features = '/files/Chloe/mVals.tsv'
file_mValues_desc = '/files/Chloe/mValues_desc.csv'
Features = hf.read_table_with_progress(file_Features)

# Transpose the methylation data so that rows = samples and columns = features
mValues = Features.iloc[:, 1:].transpose()
print("Data loaded and transposed successfully.")

# Load Descriptors and filter out all columns with std = 0
mValues_desc = pd.read_csv(file_mValues_desc, sep=';')
mValues = mValues.loc[:, np.array(mValues_desc["std"] != 0)]
print("Shape of mValues (rows x columns):", mValues.shape)

# Load demographics and align with methylation data
file_Demographics = "/files/Chloe/Demographics.xlsx"
Demographics = pd.read_excel(file_Demographics)
Demographics.set_index('index', inplace=True)

# Ensure indices match for m values
mValues.index = mValues.index.str.strip()
mValues_aligned = mValues.reindex(Demographics.index).dropna()
Demographics = Demographics.loc[mValues_aligned.index]

# Prepare feature matrix (X) and label array (y)
X_raw = mValues_aligned.values
y_raw = Demographics['ADHD.average'].values

print("Final feature matrix shape:", X_raw.shape)
print("Label array shape:", y_raw.shape)

# Remove any rows with NaN values in y
X_raw = X_raw[~np.isnan(y_raw)]
y_raw = y_raw[~np.isnan(y_raw)]

print(f"After removing NaN in y, we have {X_raw.shape[0]} samples.")

# %% STEP 1: Split Data into Train and Test Set
X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
    X_raw,  
    y_raw,          
    test_size=0.2,  
    shuffle=True,   
    random_state=40 
)
print("STEP 1: Train/Test Split Done.")
print("Train+Validation Set Shape:", X_trainval.shape)
print("Test Set Shape:", X_test_final.shape)

# %% STEP 2: Feature Selection Based on Correlation Threshold
correlation_threshold = 0.2
correlations = np.array([np.corrcoef(X_trainval[:, i], y_trainval)[0, 1] for i in range(X_trainval.shape[1])])
selected_features = np.abs(correlations) >= correlation_threshold

X_trainval_selected = X_trainval[:, selected_features]
X_test_final_selected = X_test_final[:, selected_features]

print("STEP 2: Feature Selection Done.")
print(f"Selected {X_trainval_selected.shape[1]} features with correlation >= {correlation_threshold}.")

# %% STEP 2: Normalize Features
scaler = MinMaxScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval_selected)
X_test_final_scaled = scaler.transform(X_test_final_selected)
print("STEP 2: Feature Normalization Done.")

# %% STEP 3: Hyperparameter Tuning for XGBoost using GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5]  # Fixed tree depth
}

# Setup the XGBoost model
xgb = XGBRegressor(random_state=42)

# Perform grid search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

# Fit the model to the training data
grid_search.fit(X_trainval_scaled, y_trainval)

# Retrieve best parameters and best score
best_params = grid_search.best_params_
best_cv_mse = -grid_search.best_score_

print(f"\nBest Parameters from GridSearchCV: {best_params}")
print(f"Best Cross-Validation MSE: {best_cv_mse:.4f}")

# %% STEP 4: Train Final XGBoost Model with Best Parameters
final_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    random_state=42
)
final_model.fit(X_trainval_scaled, y_trainval)

print("\nSTEP 4: Final XGBoost Model Trained.")

# %% STEP 5: Evaluate Model on Final Test Set
y_test_pred = final_model.predict(X_test_final_scaled)

mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)

print("STEP 5: Final Test Set Performance")
print(f"  Features Selected: {X_trainval_selected.shape[1]}")
print(f"  MSE (test): {mse_test:.4f}")
print(f"  R2 (test): {r2_test:.4f}")

# %%
# %% Step 6: Plot Predictions vs. Actuals
plt.figure(figsize=(8, 6))
plt.scatter(y_test_final, y_test_pred, alpha=0.7, color='teal')
plt.plot([min(y_test_final), max(y_test_final)], [min(y_test_final), max(y_test_final)], 'r--')
plt.xlabel('Actual ADHD Scores')
plt.ylabel('Predicted ADHD Scores')
plt.title('Predicted vs. Actual ADHD Scores')
plt.grid(True)
plt.show()
# %%
