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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# %% Step 0: Data Loading and Preprocessing
file_Features = '/files/Chloe/mVals.tsv'
file_mValues_desc = '/files/Chloe/mValues_desc.csv'
Features = hf.read_table_with_progress(file_Features)

mValues = Features.iloc[:, 1:].transpose()
print("Data loaded and transposed successfully.")

mValues_desc = pd.read_csv(file_mValues_desc, sep=';')
mValues = mValues.loc[:, np.array(mValues_desc["std"] != 0)]
print("Shape of mValues (rows x columns):", mValues.shape)

file_Demographics = "/files/Chloe/Demographics.xlsx"
Demographics = pd.read_excel(file_Demographics)
Demographics.set_index('index', inplace=True)

mValues.index = mValues.index.str.strip()
mValues_aligned = mValues.reindex(Demographics.index).dropna()
Demographics = Demographics.loc[mValues_aligned.index]

X_raw = mValues_aligned.values
y_raw = Demographics['ADHD.average'].values

valid_indices = ~np.isnan(y_raw)
X_raw = X_raw[valid_indices]
y_raw = y_raw[valid_indices]

print(f"After removing NaN in y, we have {X_raw.shape[0]} samples.")

# %% Step 1: Train-Test Split
X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
    X_raw, y_raw, test_size=0.2, shuffle=True, random_state=40
)
print("Train/Test Split Done.")

# %% Step 2: PCA Cross-Validation to Select Optimal Components (max 20 components)
kf = KFold(n_splits=5, shuffle=True, random_state=43)
pca_components = [1, 2, 5, 10, 15, 20]  # Set max components to 20

best_cv_mse, best_n_components = float("inf"), None

for n in pca_components:
    cv_mse_scores, cv_r2_scores = [], []
    for train_index, val_index in kf.split(X_trainval):
        X_cv_train, X_cv_val = X_trainval[train_index], X_trainval[val_index]
        y_cv_train, y_cv_val = y_trainval[train_index], y_trainval[val_index]
        
        scaler = MinMaxScaler()
        X_cv_train_scaled = scaler.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler.transform(X_cv_val)
        
        pca = PCA(n_components=n)
        X_cv_train_pca = pca.fit_transform(X_cv_train_scaled)
        X_cv_val_pca = pca.transform(X_cv_val_scaled)

        model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        model.fit(X_cv_train_pca, y_cv_train)
        
        y_cv_pred = model.predict(X_cv_val_pca)
        cv_mse_scores.append(mean_squared_error(y_cv_val, y_cv_pred))
        cv_r2_scores.append(r2_score(y_cv_val, y_cv_pred))

    mse_avg, r2_avg = np.mean(cv_mse_scores), np.mean(cv_r2_scores)
    print(f"PCA Components: {n} | CV Avg MSE: {mse_avg:.4f} | CV Avg R2: {r2_avg:.4f}")
    
    if mse_avg < best_cv_mse:
        best_cv_mse, best_n_components = mse_avg, n

print(f"\nBest PCA Components from CV: {best_n_components} | Best CV MSE: {best_cv_mse:.4f}")

# %% Step 3: Transform Train and Test Set with a Max of 20 PCA Components
scaler_full = MinMaxScaler()
X_trainval_scaled = scaler_full.fit_transform(X_trainval)
pca_full = PCA(n_components=min(best_n_components, 20))  # Ensure we do not exceed 20 components
X_trainval_pca = pca_full.fit_transform(X_trainval_scaled)

X_test_final_scaled = scaler_full.transform(X_test_final)
X_test_final_pca = pca_full.transform(X_test_final_scaled)

print("Train and Test Sets Transformed.")

# %% Step 4: Hyperparameter Tuning for XGBoost using GridSearchCV
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
grid_search.fit(X_trainval_pca, y_trainval)

# Retrieve best parameters and best score
best_params = grid_search.best_params_
best_cv_mse = -grid_search.best_score_

print(f"\nBest Parameters from GridSearchCV: {best_params}")
print(f"Best Cross-Validation MSE: {best_cv_mse:.4f}")

# %% Step 5: Train Final XGBoost Model with Best Parameters
final_model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    random_state=42
)
final_model.fit(X_trainval_pca, y_trainval)
print("Final XGBoost Model Trained.")

# %% Step 6: Test Set Prediction
y_test_pred = final_model.predict(X_test_final_pca)

# %% Step 7: Evaluate Model on Final Test Set
mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)

print(f"\nFinal Test Set Performance")
print(f"Best PCA component: {best_n_components}")
print(f"MSE (test): {mse_test:.4f}")
print(f"R2 (test): {r2_test:.4f}")

# %% Step 8: Learning Curve
train_sizes = np.linspace(0.1, 0.9, 9)
train_mse, val_mse = [], []

for size in train_sizes:
    X_subset, _, y_subset, _ = train_test_split(X_trainval_pca, y_trainval, train_size=size, random_state=42)
    final_model.fit(X_subset, y_subset)
    train_mse.append(mean_squared_error(y_subset, final_model.predict(X_subset)))
    val_mse.append(mean_squared_error(y_test_final, final_model.predict(X_test_final_pca)))

plt.figure(figsize=(8, 6))
plt.plot(train_sizes * 100, train_mse, 'o-', label='Training MSE')
plt.plot(train_sizes * 100, val_mse, 'o-', label='Validation MSE')
plt.title('Learning Curve')
plt.xlabel('Training Set Size (%)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()


# %% Step 9: Plot Predictions vs. Actuals
plt.figure(figsize=(8, 6))
plt.scatter(y_test_final, y_test_pred, alpha=0.7, color='teal')
plt.plot([min(y_test_final), max(y_test_final)], [min(y_test_final), max(y_test_final)], 'r--')
plt.xlabel('Actual ADHD Scores')
plt.ylabel('Predicted ADHD Scores')
plt.title('Predicted vs. Actual ADHD Scores')
plt.grid(True)
plt.show()

# %%
