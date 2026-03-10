import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

print("Original dataset shape:", df.shape)

# Remove any unnamed columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Separate features and target
X = df.drop(['id', 'diagnosis'], axis=1)
y = df['diagnosis']

print(f"Original number of features: {X.shape[1]}")

# Handle missing values using mean imputation
print(f"Rows with missing values: {X.isna().any(axis=1).sum()}")
print(f"Total NaN values: {X.isna().sum().sum()}")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)
print(f"Dataset shape after imputation: {X.shape}")

# Step 1: Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data normalized using StandardScaler")

# Step 2: Reduce features to 5 components
pca = PCA(n_components=5)
X_reduced = pca.fit_transform(X_scaled)

print(f"\nPCA - Number of components: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
print(f"Total variance retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

# Create DataFrame with reduced features
reduced_features = [f'PC{i+1}' for i in range(X_reduced.shape[1])]
X_reduced_df = pd.DataFrame(X_reduced, columns=reduced_features)
X_reduced_df['diagnosis'] = y.values
X_reduced_df['id'] = df['id'].values

# Save the processed data
output_file = 'data_normalized_reduced.csv'
X_reduced_df.to_csv(output_file, index=False)
print(f"\n✓ Processed data saved to '{output_file}'")

# Visualize explained variance
plt.figure(figsize=(10, 6))
components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, pca.explained_variance_ratio_, alpha=0.7, label='Individual Variance')
plt.plot(components, np.cumsum(pca.explained_variance_ratio_), 'ro-', linewidth=2, label='Cumulative Variance')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Variance Explained by Each Component')
plt.xticks(components)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=150)
print("✓ PCA variance plot saved to 'pca_variance.png'")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Original features: {X.shape[1]}")
print(f"Reduced features: {X_reduced.shape[1]}")
print(f"Reduction: {X.shape[1]} → {X_reduced.shape[1]} ({100*X_reduced.shape[1]/X.shape[1]:.1f}%)")
print(f"Variance retained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
print("\nFirst 5 rows of reduced dataset:")
print(X_reduced_df.head())