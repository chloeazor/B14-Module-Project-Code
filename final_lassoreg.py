#%% Import all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/files/Chloe')

import help_functions as hf
hf.install_packages()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

#%% Step 0: Data preprocessing
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

print("Final feature matrix shape:", X_raw.shape)
print("Label array shape:", y_raw.shape)

valid_indices = ~np.isnan(y_raw)
X_raw = X_raw[valid_indices]
y_raw = y_raw[valid_indices]

print(f"After removing NaN in y, we have {X_raw.shape[0]} samples.")

# %% STEP 1: Split Data into Train and Test Set
X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
    X_raw, y_raw, test_size=0.2, shuffle=True, random_state=40
)
print("STEP 1: Train/Test Split Done.")
print("Train+Validation size:", X_trainval.shape[0])
print("Test set size:", X_test_final.shape[0])

# %% STEP 2: PCA Hyperparameter Tuning (Max 20 Components)
kf = KFold(n_splits=5, shuffle=True, random_state=43)

max_pca_components = min(20, X_trainval.shape[1])
pca_components = [1, 2, 5, 10, max_pca_components]

best_cv_mse = float("inf")
best_cv_r2 = float("-inf")
best_n_components = None
list_of_average_MSE = []

for n in pca_components:
    cv_mse_scores = []
    cv_r2_scores = []

    for train_index, cv_val_index in kf.split(X_trainval):
        X_cv_train, X_cv_val = X_trainval[train_index], X_trainval[cv_val_index]
        y_cv_train, y_cv_val = y_trainval[train_index], y_trainval[cv_val_index]

        scaler_cv = MinMaxScaler()
        X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler_cv.transform(X_cv_val)

        pca_cv = PCA(n_components=n)
        X_cv_train_pca = pca_cv.fit_transform(X_cv_train_scaled)
        X_cv_val_pca = pca_cv.transform(X_cv_val_scaled)

        model_cv = Lasso(alpha=0.1)
        model_cv.fit(X_cv_train_pca, y_cv_train)

        y_cv_pred = model_cv.predict(X_cv_val_pca)
        mse_cv = mean_squared_error(y_cv_val, y_cv_pred)
        r2_cv = r2_score(y_cv_val, y_cv_pred)

        cv_mse_scores.append(mse_cv)
        cv_r2_scores.append(r2_cv)

    mse_avg = np.mean(cv_mse_scores)
    r2_avg = np.mean(cv_r2_scores)
    list_of_average_MSE.append(mse_avg)

    print(f"PCA Components: {n} | CV Avg MSE: {mse_avg:.4f} | CV Avg R2: {r2_avg:.4f}")

    if mse_avg < best_cv_mse:
        best_cv_mse = mse_avg
        best_cv_r2 = r2_avg
        best_n_components = n

print(f"\nBest PCA Components from CV (capped at 20): {min(best_n_components, 20)} | Best CV MSE: {best_cv_mse:.4f} | Best CV R2: {best_cv_r2:.4f}")

# %% STEP 3: Apply PCA with the best number of components (capped at 20)
scaler_full = MinMaxScaler()
X_trainval_scaled = scaler_full.fit_transform(X_trainval)
pca_full = PCA(n_components=min(best_n_components, 20))
X_trainval_pca = pca_full.fit_transform(X_trainval_scaled)

X_test_final_scaled = scaler_full.transform(X_test_final)
X_test_final_pca = pca_full.transform(X_test_final_scaled)

print("STEP 3: Train and Test Sets Transformed Using Best PCA Components.")

# %% STEP 4: Hyperparameter Search for Lasso
alphas = np.logspace(-4, 1, 10)
best_alpha = None
best_mse = float("inf")

for alpha in alphas:
    lasso_cv = Lasso(alpha=alpha)
    mse_scores = []

    for train_index, cv_val_index in kf.split(X_trainval_pca):
        X_cv_train, X_cv_val = X_trainval_pca[train_index], X_trainval_pca[cv_val_index]
        y_cv_train, y_cv_val = y_trainval[train_index], y_trainval[cv_val_index]

        lasso_cv.fit(X_cv_train, y_cv_train)
        y_cv_pred = lasso_cv.predict(X_cv_val)
        mse_scores.append(mean_squared_error(y_cv_val, y_cv_pred))

    avg_mse = np.mean(mse_scores)
    print(f"Alpha: {alpha:.5f} | CV Avg MSE: {avg_mse:.4f}")

    if avg_mse < best_mse:
        best_mse = avg_mse
        best_alpha = alpha

print(f"\nBest Alpha for Lasso: {best_alpha:.5f} | Best CV MSE: {best_mse:.4f}")

# %% STEP 5: Train Final Lasso Model
final_model = Lasso(alpha=best_alpha)
final_model.fit(X_trainval_pca, y_trainval)
print("\nSTEP 5: Final Lasso Model Trained.")

# %% STEP 6 & 7: Predict and Evaluate on Test Set
y_test_pred = final_model.predict(X_test_final_pca)
mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)

print(f"\nSTEP 7: Final Test Set Performance")
print(f"  Best PCA Components: {min(best_n_components, 20)}")
print(f"  Best Lasso Alpha: {best_alpha:.5f}")
print(f"  MSE (test): {mse_test:.4f}")
print(f"  R2 (test): {r2_test:.4f}")

# %% Step 8: Plot PCA Components vs. Average MSE
plt.figure(figsize=(8, 5))
plt.plot(pca_components, list_of_average_MSE, marker='o', linestyle='--', color='b')
plt.xlabel('Number of PCA Components')
plt.ylabel('Average MSE')
plt.title('PCA Components vs. Average MSE')
plt.xlim(1, max(pca_components))
plt.grid()
plt.show()

# %% Step 9: Residuals Plot
residuals = y_test_final - y_test_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='r')
plt.axvline(0, color='black', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.grid()
plt.show()

# %% Step 10: Predictions vs. Actuals Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test_final, y=y_test_pred, alpha=0.7, edgecolors='k')
plt.plot([min(y_test_final), max(y_test_final)], [min(y_test_final), max(y_test_final)], linestyle='--', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual Values')
plt.grid()
plt.show()

# %%
