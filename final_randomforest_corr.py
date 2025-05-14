# %% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/files/Chloe')

import help_functions as hf
hf.install_packages()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %% Step 0: Load and preprocess data
file_Features = '/files/Chloe/mVals.tsv'
file_mValues_desc = '/files/Chloe/mValues_desc.csv'
Features = hf.read_table_with_progress(file_Features)

# Transpose methylation data
mValues = Features.iloc[:, 1:].transpose()
print("Data loaded and transposed successfully.")

# Load descriptors and filter zero-variance features
mValues_desc = pd.read_csv(file_mValues_desc, sep=';')
mValues = mValues.loc[:, np.array(mValues_desc["std"] != 0)]
print("Shape of mValues (rows x columns):", mValues.shape)

# Load and align demographics
file_Demographics = "/files/Chloe/Demographics.xlsx"
Demographics = pd.read_excel(file_Demographics)
Demographics.set_index('index', inplace=True)

mValues.index = mValues.index.str.strip()
mValues_aligned = mValues.reindex(Demographics.index).dropna()
Demographics = Demographics.loc[mValues_aligned.index]

# Prepare X and y
X_raw = mValues_aligned.values
y_raw = Demographics['ADHD.average'].values

# Remove NaN targets
valid_indices = ~np.isnan(y_raw)
X_raw = X_raw[valid_indices]
y_raw = y_raw[valid_indices]

print(f"After removing NaN in y, we have {X_raw.shape[0]} samples.")

# %% Step 1: Split data into training/validation and test sets
X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
    X_raw,
    y_raw,
    test_size=0.2,
    shuffle=True,
    random_state=40
)
print("STEP 1: Train/Test Split Done.")
print("Train+Validation size:", X_trainval.shape[0])
print("Test set size:", X_test_final.shape[0])

# %% Step 2: Feature selection based on correlation threshold
correlations = np.array([np.corrcoef(X_trainval[:, i], y_trainval)[0, 1] for i in range(X_trainval.shape[1])])
selected_features = np.abs(correlations) >= 0.2

# Apply feature selection
X_trainval_selected = X_trainval[:, selected_features]
X_test_final_selected = X_test_final[:, selected_features]

print("STEP 2: Feature Selection Done.")
print(f"Selected {X_trainval_selected.shape[1]} features out of {X_trainval.shape[1]}")

# Normalize features
scaler = MinMaxScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval_selected)
X_test_final_scaled = scaler.transform(X_test_final_selected)

# %% Step 3: Hyperparameter tuning using GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=43)

param_grid = {
    "n_estimators": [50, 100, 200],
    "min_samples_split": [2, 5, 10],
    "max_depth": [5]  # Fixed tree depth
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=kf,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_trainval_scaled, y_trainval)

best_params = grid_search.best_params_
best_mse = -grid_search.best_score_

print(f"\nBest Parameters from GridSearchCV: {best_params}")
print(f"Best Cross-Validation MSE: {best_mse:.4f}")

# %% Step 4: Train final Random Forest model
final_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42
)
final_model.fit(X_trainval_scaled, y_trainval)

print("\nSTEP 4: Final Random Forest Model Trained.")

# %% Step 5: Apply model to the test set
y_test_pred = final_model.predict(X_test_final_scaled)

# %% Step 6: Evaluate model on the test set
mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)

print(f"\nSTEP 6: Final Test Set Performance")
print(f"  Features Selected: {X_trainval_selected.shape[1]}")
print(f"  MSE (test): {mse_test:.4f}")
print(f"  R2 (test): {r2_test:.4f}")

#
# %%
# %% Step 7: Plot Predictions vs. Actuals
plt.figure(figsize=(8, 6))
plt.scatter(y_test_final, y_test_pred, alpha=0.7, color='teal')
plt.plot([min(y_test_final), max(y_test_final)], [min(y_test_final), max(y_test_final)], 'r--')
plt.xlabel('Actual ADHD Scores')
plt.ylabel('Predicted ADHD Scores')
plt.title('Predicted vs. Actual ADHD Scores')
plt.grid(True)
plt.show()
# %%
