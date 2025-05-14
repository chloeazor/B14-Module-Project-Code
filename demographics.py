
#%% Import necessary libraries
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

#%% Data import and preprocessing
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

# Check the first few rows of Demographics to understand structure
print(Demographics.head())

# Basic stats
num_children = Demographics.shape[0]
age_min = Demographics['Age'].min()
age_max = Demographics['Age'].max()
age_mean = Demographics['Age'].mean()

print(f"Number of children: {num_children}")
print(f"Age range: {age_min} - {age_max} (Mean: {age_mean:.2f})")

# Gender distribution
gender_counts = Demographics['Gender'].value_counts()
print("\nGender Distribution:")
print(gender_counts)

# Set up the style
sns.set(style="whitegrid")

# Histogram of Age
plt.figure(figsize=(10, 5))
sns.histplot(Demographics['Age'], bins=10, kde=True, color='skyblue')
plt.title('Age Distribution of Children')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Bar chart of Gender distribution
plt.figure(figsize=(7, 4))
sns.countplot(x='Gender', data=Demographics, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
####### demographics after NaN have been removed

# Subset demographics to match valid samples
Demographics_valid = Demographics[valid_indices]

# Now use Demographics_valid instead of Demographics
print(Demographics_valid.head())

# Basic stats
num_children = Demographics_valid.shape[0]
age_min = Demographics_valid['Age'].min()
age_max = Demographics_valid['Age'].max()
age_mean = Demographics_valid['Age'].mean()

print(f"Number of children: {num_children}")
print(f"Age range: {age_min} - {age_max} (Mean: {age_mean:.2f})")

# Gender distribution
gender_counts = Demographics_valid['Gender'].value_counts()
print("\nGender Distribution:")
print(gender_counts)

# Set up the style
sns.set(style="whitegrid")

# Histogram of Age
plt.figure(figsize=(10, 5))
sns.histplot(Demographics_valid['Age'], bins=10, kde=True, color='skyblue')
plt.title('Age Distribution of Children')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Bar chart of Gender distribution
plt.figure(figsize=(7, 4))
sns.countplot(x='Gender', data=Demographics_valid, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
# %% Visualizations of the Gender and Age distribution
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(style="whitegrid")

# Create a figure with 2 subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age Distribution (Histogram with KDE)
sns.histplot(Demographics['Age'], bins=10, kde=True, color='skyblue', ax=axes[0])
axes[0].set_title('Age Distribution')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Gender Distribution (Bar chart)
sns.countplot(x='Gender', data=Demographics, palette='pastel', ax=axes[1])
axes[1].set_title('Gender Distribution')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Count')

# Tight layout for better spacing
plt.tight_layout()
plt.show()

# %% Participant specifics, min and max age, gender counts, ADHD severity ranges.
# Number of children
num_children = Demographics.shape[0]

# Age stats
age_min = Demographics['Age'].min()
age_max = Demographics['Age'].max()
age_mean = Demographics['Age'].mean()

# Gender distribution
gender_counts = Demographics['Gender'].value_counts()

# Print output
print("----- Demographics Summary -----")
print(f"Total number of children: {num_children}")
print(f"Age range: {age_min} - {age_max} years")
print(f"Average age: {age_mean:.2f} years\n")

print("Gender distribution:")
for gender, count in gender_counts.items():
    print(f"  {gender}: {count} children")

# Severity column
severity_col = 'ADHD.average'

# Basic stats
severity_min = Demographics[severity_col].min()
severity_max = Demographics[severity_col].max()
severity_mean = Demographics[severity_col].mean()

print("----- ADHD Severity Summary -----")
print(f"Severity range: {severity_min:.2f} - {severity_max:.2f}")
print(f"Average severity: {severity_mean:.2f}")

# %% Plot showing the optimal correlation threshold that should be used as a feature reduction strategy
#import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Compute absolute Pearson correlation of each feature with the target
correlations = np.array([abs(pearsonr(X_raw[:, i], y_raw)[0]) for i in range(X_raw.shape[1])])

# Define discrete correlation thresholds
thresholds = [0, 0.1, 0.2, 0.3, 0.4]
features_retained = [(correlations >= t).sum() for t in thresholds]

# Plot
plt.figure(figsize=(7, 5))
plt.plot(thresholds, features_retained, marker='o', linestyle='-')
plt.xlabel('Correlation Threshold')
plt.ylabel('Number of Features Retained')
plt.title('Features Retained at Different Correlation Thresholds')
plt.xticks(thresholds)
plt.grid(True)
plt.tight_layout()
plt.show()


# %% Plot showing the optimal number of PCA components to be used as a feature reduction strategy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming X_scaled is your scaled feature matrix
pca = PCA()
pca.fit(X_scaled)

# Limit to first 100 components
explained_variance = pca.explained_variance_ratio_[:100]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, 101), explained_variance, marker='o', linestyle='-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# %%
