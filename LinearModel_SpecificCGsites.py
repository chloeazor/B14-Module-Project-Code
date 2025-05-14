#%% Import Libraries
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#%% Step 0: Data Preprocessing - Load Methylation Data, Descriptors, and Demographics
file_Features = '/files/Chloe/mVals.tsv'
file_mValues_desc = '/files/Chloe/mValues_desc.csv'
Features = hf.read_table_with_progress(file_Features)

# Transpose methylation data: rows = samples, columns = features
mValues = Features.iloc[:, 1:].transpose()
print("Data loaded and transposed successfully.")

# Load mValues_desc to filter specific CpG sites
mValues_desc = pd.read_csv(file_mValues_desc, sep=';')

# CpG sites to keep
target_cpg_sites = [
    'cg17478313', 'cg21609804', 'cg03416665', 
    'cg27034450', 'cg02466711', 'cg00964221', 'cg06972911'
]

# mValues_desc first column has no name, access it by position
mValues_desc_cpgs = mValues_desc.iloc[:, 0]

# Filter columns based on target CpG sites
filtered_cols = [i for i, cpg in enumerate(mValues_desc_cpgs) if cpg in target_cpg_sites]
mValues_filtered = mValues.iloc[:, filtered_cols]
print(f"Filtered mValues to include only target CpG sites: {mValues_filtered.shape}")

# Load demographics and align
file_Demographics = "/files/Chloe/Demographics.xlsx"
Demographics = pd.read_excel(file_Demographics)
Demographics.set_index('index', inplace=True)

# Align indices
mValues_filtered.index = mValues_filtered.index.str.strip()
mValues_aligned = mValues_filtered.reindex(Demographics.index).dropna()
Demographics = Demographics.loc[mValues_aligned.index]

# Prepare feature matrix and label
X_raw = mValues_aligned.values
y_raw = Demographics['ADHD.average'].values

# Remove NaNs in y
valid_indices = ~np.isnan(y_raw)
X_raw = X_raw[valid_indices]
y_raw = y_raw[valid_indices]

print(f"Final data shapes - X: {X_raw.shape}, y: {y_raw.shape}")

#%% STEP 1: Train-Test Split
X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
    X_raw, y_raw, test_size=0.2, shuffle=True, random_state=40
)
print("STEP 1: Train/Test Split Done.")

#%% STEP 2: PCA Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=43)
pca_components = [1, 2, 3, 5, 7]  # Adjusted since we have 7 features max

best_cv_mse = float("inf")
best_n_components = None
list_of_average_MSE = []

for n in pca_components:
    cv_mse_scores = []
    for train_idx, val_idx in kf.split(X_trainval):
        X_cv_train, X_cv_val = X_trainval[train_idx], X_trainval[val_idx]
        y_cv_train, y_cv_val = y_trainval[train_idx], y_trainval[val_idx]

        scaler_cv = MinMaxScaler()
        X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler_cv.transform(X_cv_val)

        pca_cv = PCA(n_components=n)
        X_cv_train_pca = pca_cv.fit_transform(X_cv_train_scaled)
        X_cv_val_pca = pca_cv.transform(X_cv_val_scaled)

        model_cv = LinearRegression()
        model_cv.fit(X_cv_train_pca, y_cv_train)
        y_cv_pred = model_cv.predict(X_cv_val_pca)

        cv_mse_scores.append(mean_squared_error(y_cv_val, y_cv_pred))

    mse_avg = np.mean(cv_mse_scores)
    list_of_average_MSE.append(mse_avg)

    print(f"PCA Components: {n} | CV Avg MSE: {mse_avg:.4f}")
    if mse_avg < best_cv_mse:
        best_cv_mse = mse_avg
        best_n_components = n

print(f"\nBest PCA Components from CV: {best_n_components} | Best CV MSE: {best_cv_mse:.4f}")

#%% STEP 3: Apply Best PCA to Train/Test
scaler_full = MinMaxScaler()
X_trainval_scaled = scaler_full.fit_transform(X_trainval)

pca_full = PCA(n_components=best_n_components)
X_trainval_pca = pca_full.fit_transform(X_trainval_scaled)

X_test_final_scaled = scaler_full.transform(X_test_final)
X_test_final_pca = pca_full.transform(X_test_final_scaled)

print("STEP 3: PCA Transformation Done.")

#%% STEP 4: Train Final Linear Regression Model
final_model = LinearRegression()
final_model.fit(X_trainval_pca, y_trainval)
print("STEP 4: Model Trained.")

#%% STEP 5: Evaluate on Test Set
y_test_pred = final_model.predict(X_test_final_pca)
mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)

print(f"\nSTEP 5: Final Test Set Performance")
print(f"MSE (test): {mse_test:.4f}")
print(f"R2 (test): {r2_test:.4f}")

#%% Optional: Plot PCA Components vs Average MSE
plt.figure(figsize=(8, 5))
plt.plot(pca_components, list_of_average_MSE, marker='o', linestyle='--')
plt.xlabel('Number of PCA Components')
plt.ylabel('Average CV MSE')
plt.title('PCA Components vs Average MSE')
plt.grid()
plt.show()

#%% Optional: Residuals Plot
residuals = y_test_final - y_test_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='r')
plt.axvline(0, color='black', linestyle='--')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.grid()
plt.show()

#%% Optional: Predictions vs. Actual Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test_final, y=y_test_pred, alpha=0.7, edgecolors='k')
plt.plot([min(y_test_final), max(y_test_final)], [min(y_test_final), max(y_test_final)], linestyle='--', color='red')
plt.xlabel('Actual ADHD.average')
plt.ylabel('Predicted ADHD.average')
plt.title('Predicted vs Actual ADHD.average')
plt.grid()
plt.show()

# %%
