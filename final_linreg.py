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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

#%% Step 0: Data preprocessing - Load Methylation data, descriptors, and demographics
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

# %% STEP 2: Use PCA to Reduce Dimensionality (Hyperparameter Tuning)
kf = KFold(n_splits=5, shuffle=True, random_state=43)

# Cap PCA components to max 20
max_pca = min(20, X_trainval.shape[1])
pca_components = [i for i in [1, 2, 5, 10, 15, 20] if i <= max_pca]

best_cv_mse = float("inf") 
best_cv_r2 = float("-inf")
best_n_components = None
list_of_average_MSE = []

for n in pca_components:
    cv_mse_scores, cv_r2_scores = [], []
    
    for train_index, cv_val_index in kf.split(X_trainval):
        X_cv_train, X_cv_val = X_trainval[train_index], X_trainval[cv_val_index]
        y_cv_train, y_cv_val = y_trainval[train_index], y_trainval[cv_val_index]
        
        scaler_cv = MinMaxScaler()
        X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
        X_cv_val_scaled = scaler_cv.transform(X_cv_val)
        
        pca_cv = PCA(n_components=n)
        X_cv_train_pca = pca_cv.fit_transform(X_cv_train_scaled)
        X_cv_val_pca = pca_cv.transform(X_cv_val_scaled)
        
        model_cv = LinearRegression()
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

print(f"\n Best PCA Components from CV: {best_n_components} | Best CV MSE: {best_cv_mse:.4f} | Best CV R2: {best_cv_r2:.4f}")

# %% STEP 3: Apply Best PCA and Scale
scaler_full = MinMaxScaler()
X_trainval_scaled = scaler_full.fit_transform(X_trainval)
pca_full = PCA(n_components=best_n_components)
X_trainval_pca = pca_full.fit_transform(X_trainval_scaled)

X_test_final_scaled = scaler_full.transform(X_test_final)
X_test_final_pca = pca_full.transform(X_test_final_scaled)

print("STEP 3: Train and Test Sets Transformed Using Best PCA Components.")

# %% STEP 4: Hyperparameter search
# Linear Regression has no hyperparameters to search

# %% STEP 5: Train Final Model on Full Train+Val Set
final_model = LinearRegression()
final_model.fit(X_trainval_pca, y_trainval)
print("\nSTEP 5: Final Linear Regression Model Trained.")

# %% STEP 6: Apply Model on Test Set
y_test_pred = final_model.predict(X_test_final_pca)

# %% STEP 7: Evaluate Model on Final Test Set
mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)

print(f"\n✅ FINAL TEST SET PERFORMANCE")
print(f"Best PCA Components: {best_n_components}")
print(f"MSE (test): {mse_test:.4f}")
print(f"R2 (test): {r2_test:.4f}")

# %% Step 8: Plot PCA Components vs. Average MSE
print("PCA Components Tested:", pca_components)
print("MSE Scores:", list_of_average_MSE)

plt.figure(figsize=(8, 5))
plt.plot(pca_components, list_of_average_MSE, marker='o', linestyle='--', color='b')
plt.xlabel('Number of PCA Components')
plt.ylabel('Average MSE')
plt.title('PCA Components vs. Average MSE')
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

# %% Print the versions of all libraries used
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import matplotlib
import seaborn as sns
import sys

print("Python version:", sys.version)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("XGBoost version:", xgb.__version__)
print("Matplotlib version:", matplotlib.__version__)
print("Seaborn version:", sns.__version__)

# %% Step 11: Statistical Significance Testing via Permutation Test
 
n_permutations = 1000          # Number of shuffled worlds to sample
perm_mse_list   = []   # Re-fit the entire preprocessing + model pipeline on permuted data
    
scaler_perm = MinMaxScaler()
X_trainval_scaled_perm = scaler_perm.fit_transform(X_trainval)

pca_perm   = PCA(n_components=best_n_components)
X_trainval_pca_perm = pca_perm.fit_transform(X_trainval_scaled_perm)         # Store one MSE per permutation

print("Running permutation test")
for i in range(n_permutations):
    # Shuffle the TRAIN+VAL labels to break any real signal
    y_trainval_perm = np.random.permutation(y_trainval)

   

    perm_model = LinearRegression()
    perm_model.fit(X_trainval_pca_perm, y_trainval_perm)

    # Transform the (untouched) external test set with the *permutation* scalers
    X_test_scaled_perm = scaler_perm.transform(X_test_final)
    X_test_pca_perm    = pca_perm.transform(X_test_scaled_perm)

    # Predict and score
    y_test_perm_pred = perm_model.predict(X_test_pca_perm)
    mse_perm = mean_squared_error(y_test_final, y_test_perm_pred)
    perm_mse_list.append(mse_perm)

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{n_permutations} permutations complete …")

perm_mse_array = np.array(perm_mse_list)

# p-value = fraction of null MSEs as low or lower than the real one
p_value = np.mean(perm_mse_array <= mse_test)

print("\n📊 Permutation-test summary")
print(f"Actual model MSE (test set): {mse_test:.4f}")
print(f"Mean permuted-model MSE:     {perm_mse_array.mean():.4f}")
print(f"P-value:                     {p_value:.4f}")




# %% Step 12: Plot Random MSE Distribution with 95% Confidence Interval

ci_low, ci_high = np.percentile(perm_mse_array, [2.5, 97.5])

plt.figure(figsize=(10, 6))
sns.histplot(perm_mse_array, bins=30, kde=True, color='skyblue',
             label='Permuted-model MSEs')
plt.axvline(perm_mse_array.mean(), color='blue', linestyle='-',
            label=f'Mean Permuted MSE ({perm_mse_array.mean():.4f})')
plt.axvline(ci_low,  color='green', linestyle='--',
            label=f'2.5 % Quantile ({ci_low:.4f})')
plt.axvline(ci_high, color='green', linestyle='--',
            label=f'97.5 % Quantile ({ci_high:.4f})')
plt.axvline(mse_test, color='red', linestyle='-', linewidth=2.5,
            label=f'Actual Model MSE ({mse_test:.4f})')

plt.fill_betweenx(y=[0, plt.gca().get_ylim()[1]],
                  x1=ci_low, x2=ci_high, color='green', alpha=0.10)
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Frequency')
plt.title('Permutation Test: Null-model MSEs vs. Actual Model MSE')
plt.legend()
plt.grid()
plt.show()
