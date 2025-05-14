# %% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

sys.path.append('/files/Chloe')
import help_functions as hf
hf.install_packages()

# %% Step 0: Data Preprocessing
file_Features = '/files/Chloe/mVals.tsv'
file_mValues_desc = '/files/Chloe/mValues_desc.csv'
Features = hf.read_table_with_progress(file_Features)

mValues = Features.iloc[:, 1:].transpose()
mValues_desc = pd.read_csv(file_mValues_desc, sep=';')
mValues = mValues.loc[:, np.array(mValues_desc["std"] != 0)]

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

# %% Step 1: Train-Test Split
X_trainval, X_test_final, y_trainval, y_test_final = train_test_split(
    X_raw, y_raw, test_size=0.2, shuffle=True, random_state=40
)

# %% Step 2: Build Pipeline and Hyperparameter Grid
pipe = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA()),
    ('rf', RandomForestRegressor(random_state=42))
])

param_grid = {
    'pca__n_components': [1, 2, 5, 10, 15, 20],   # Max PCA components = 20
    'rf__n_estimators': [50, 100, 200],
    'rf__min_samples_split': [2, 5, 10],
    'rf__max_depth': [5],  # Fixed tree depth
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)

# %% Step 3: Run GridSearchCV
grid_search.fit(X_trainval, y_trainval)

print("\nBest parameters found:", grid_search.best_params_)
print(f"Best CV MSE: {-grid_search.best_score_:.4f}")

# %% Step 4: Apply best model to Test Set
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test_final)

# %% Step 5: Evaluate Test Set Performance
mse_test = mean_squared_error(y_test_final, y_test_pred)
r2_test = r2_score(y_test_final, y_test_pred)

print("\nBest parameters found:", grid_search.best_params_)
print(f"  MSE (test): {mse_test:.4f}")
print(f"  R2 (test): {r2_test:.4f}")


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
