###EDA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode
import warnings
warnings.filterwarnings("ignore")

# Load dataset
file_path = "6156538.txt"
column_names = [
    "Pelvic Incidence", "Pelvic Tilt", "Lumbar Lordosis Angle",
    "Sacral Slope", "Pelvic Radius", "Grade of Spondylolisthesis", "Class"
]
df = pd.read_csv(file_path, sep=" ", header=None, names=column_names, engine="python")
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df["Class"] = df["Class"].map({"NO": 0, "AB": 1})

df.describe()

# Check if any non-numeric values exist in the numerical columns
non_numeric_counts = df.iloc[:, :-1].apply(lambda col: col.apply(lambda x: isinstance(x, str)).sum())
non_numeric_counts

# Check for missing values in the dataset
missing_values = df.isnull().sum()

# Display the count of missing values per column
missing_values


# Compute correlation matrix
correlation_matrix = df.iloc[:, :-1].corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Visualizing feature distributions for each class using violin plots
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    sns.violinplot(x="Class", y=col, data=df, palette="Set2")
    plt.title(f"Distribution of {col} by Class")

plt.tight_layout()
plt.show()

# Pairplot to visualize relationships between features, colored by class
sns.pairplot(df, hue="Class", diag_kind="hist", corner=True, palette="Set1")
plt.show()

features = ["Pelvic Incidence", "Pelvic Tilt", "Lumbar Lordosis Angle","Sacral Slope", "Pelvic Radius", "Grade of Spondylolisthesis"]

# Plot histograms for the most important features affecting class type
plt.figure(figsize=(15, 6))

for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df, x=feature, hue="Class", bins=30, kde=True, palette={0: "blue", 1: "red"}, alpha=0.5)
    plt.title(f"Distribution of {feature} by Class")
    plt.xlabel(feature)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

"""###PCA"""

# Separate features and target
X = df.iloc[:, :-1]
y = df["Class"]


# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[features])

# PCA to Reduce to 2 Components
pca = PCA(n_components=2)
X_pca_reduced = pca.fit_transform(X_scaled)

# Get PCA component contributions to each principal component
pca_component_contributions = pd.DataFrame(pca.components_, columns=features, index=[f"PCA{i+1}" for i in range(2)])

# Visualizing PCA Component Contributions
plt.figure(figsize=(12, 8))

for i in range(2):
    plt.subplot(1, 2, i + 1)
    sns.barplot(x=pca_component_contributions.columns, y=pca_component_contributions.iloc[i], palette="Blues_r")
    plt.title(f"Feature Contributions to PCA Component {i+1}")
    plt.xticks(rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Contribution")

plt.tight_layout()
plt.show()

"""###Unsupervised Learning"""

# Kernel K-Means with PCA
best_kernel_feature = Nystroem(gamma=1, n_components=150, random_state=42)
best_kernel_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kernel_kmeans_pipeline = make_pipeline(best_kernel_feature, best_kernel_kmeans)
best_clusters = kernel_kmeans_pipeline.fit_predict(X_pca_reduced)

# Kernel K-Means with PCA
best_clusters_original = kernel_kmeans_pipeline.fit_predict(X_scaled)

# Compute Adjusted Rand Index (ARI) to measure clustering performance
kernel_kmeans_ari = adjusted_rand_score(y, best_clusters)
kernel_kmeans_ari_original = adjusted_rand_score(y, best_clusters_original)

# Compute clustering accuracy by mapping clusters to actual class labels
# Since clustering labels are arbitrary, find the best mapping using majority voting
# Compute clustering accuracy by mapping clusters to actual class labels
def compute_cluster_accuracy(true_labels, cluster_labels):
    mapped_clusters = cluster_labels.copy()
    for cluster_label in set(cluster_labels):
        mode_result = mode(true_labels[cluster_labels == cluster_label])
        most_common_class = mode_result.mode[0] if mode_result.mode.ndim > 0 else mode_result.mode
        mapped_clusters[cluster_labels == cluster_label] = most_common_class
    return accuracy_score(true_labels, mapped_clusters)

# Compute accuracy of Kernel K-Means
kernel_kmeans_accuracy = compute_cluster_accuracy(y, best_clusters)
kernel_kmeans_accuracy_original = compute_cluster_accuracy(y, best_clusters_original)

# Display results
kernel_kmeans_ari, kernel_kmeans_accuracy

# Gaussian Mixture Model (GMM - EM Algorithm)
gmm = GaussianMixture(n_components=2, random_state=42)
gmm_clusters = gmm.fit_predict(X_pca_reduced)
gmm_clusters_original = gmm.fit_predict(X_scaled)

# Hierarchical Clustering (Agglomerative)
hierarchical = AgglomerativeClustering(n_clusters=2)
hierarchical_clusters = hierarchical.fit_predict(X_pca_reduced)
hierarchical_clusters_original = hierarchical.fit_predict(X_scaled)

# Compute ARI for each clustering method
gmm_ari = adjusted_rand_score(y, gmm_clusters)
gmm_ari_original = adjusted_rand_score(y, gmm_clusters_original)
hierarchical_ari = adjusted_rand_score(y, hierarchical_clusters)
hierarchical_ari_original = adjusted_rand_score(y, hierarchical_clusters_original)

gmm_accuracy = compute_cluster_accuracy(y, gmm_clusters)
gmm_accuracy_original = compute_cluster_accuracy(y, gmm_clusters_original)
hierarchical_accuracy = compute_cluster_accuracy(y, hierarchical_clusters)
hierarchical_accuracy_original = compute_cluster_accuracy(y, hierarchical_clusters_original)

clustering_comparison = pd.DataFrame({
    "Clustering Method": ["Kernel K-Means", "Gaussian Mixture Model (GMM)", "Hierarchical Clustering"],
    "ARI Score (With PCA)": [kernel_kmeans_ari, gmm_ari, hierarchical_ari],
    "Clustering Accuracy (With PCA)": [kernel_kmeans_accuracy, gmm_accuracy, hierarchical_accuracy],
    "ARI Score (Without PCA)": [kernel_kmeans_ari_original, gmm_ari_original, hierarchical_ari_original],
    "Clustering Accuracy (Without PCA)": [kernel_kmeans_accuracy_original, gmm_accuracy_original, hierarchical_accuracy_original]
})

# Display comparison results
display(clustering_comparison)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Define distinct colors for clarity
cluster_palette = {0: "blue", 1: "red"}
class_palette = {0: "red", 1: "blue"}
cluster_markers = {0: "o", 1: "s"}  # Dots for one cluster, squares for another
class_markers = {0: "s", 1: "o"}  # Dots for one class, squares for another

# Plot Kernel K-Means Clustering Results with corrected colors
sns.scatterplot(x=X_pca_reduced[:, 0], y=X_pca_reduced[:, 1], hue=best_clusters,
                palette=cluster_palette, style=best_clusters,
                markers=cluster_markers, alpha=0.7, ax=axes[0])
axes[0].set_title("Kernel K-Means Clustering")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

# Manually setting legend labels to match color mapping
handles_clusters, _ = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles_clusters, title="Cluster", labels=["Cluster 0", "Cluster 1"])

# Plot Actual Class Labels with corrected colors
sns.scatterplot(x=X_pca_reduced[:, 0], y=X_pca_reduced[:, 1], hue=y, palette=class_palette,
                style=y, markers=class_markers, alpha=0.7, ax=axes[1])
axes[1].set_title("Actual Class Labels")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

# Manually setting legend labels to match color mapping
handles_classes, _ = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles_classes, title="Class", labels=["Normal (0)", "Abnormal (1)"])

plt.tight_layout()
plt.show()



"""###Supervised Learning"""

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# PCA to Reduce to 4 Components
pca = PCA(n_components=4)
X_pca_reduced_4 = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca_reduced, y, test_size=0.3, random_state=42)
X_train_pca4, X_test_pca4, y_train, y_test = train_test_split(X_pca_reduced_4, y, test_size=0.3, random_state=42)

def display_metrics(clf_report):
    print("Classification Report:")
    # Print report dict as a table
    df_report = pd.DataFrame(clf_report).transpose()
    display(df_report)

model_results = {}
def fit_model(model, model_name, X_train, y_train, X_test, y_test, dtype):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print(f"{model_name} Classifier:")
    report = classification_report(y_test, y_pred, output_dict=True)
    # display_metrics(report)
    if model_name not in model_results:
        model_results[model_name] = {}
    model_results[model_name][dtype] = {
        "Precision": "{:.3f}".format(report["weighted avg"]["precision"]),
        "Accuracy": "{:.3f}".format(accuracy_score(y_test, y_pred))
    }

# Naive Bayes Classifier
nb_clf_1 = GaussianNB(var_smoothing=0.09)
fit_model(nb_clf_1, "Naive Bayes", X_train, y_train, X_test, y_test, dtype='Without PCA')

nb_clf_2 = GaussianNB(var_smoothing=0.09)
fit_model(nb_clf_2, "Naive Bayes", X_train_pca, y_train, X_test_pca, y_test, dtype='With PCA')

nb_clf_3 = GaussianNB(var_smoothing=0.09)
fit_model(nb_clf_3, "Naive Bayes", X_train_pca4, y_train, X_test_pca4, y_test, dtype='With PCA4')

# Random Forest Classifier
rf_clf_1 = RandomForestClassifier(n_estimators=100, random_state=42)
fit_model(rf_clf_1, "Random Forest", X_train, y_train, X_test, y_test, dtype='Without PCA')

rf_clf_2 = RandomForestClassifier(n_estimators=100, random_state=42)
fit_model(rf_clf_2, "Random Forest", X_train_pca, y_train, X_test_pca, y_test, dtype='With PCA')

rf_clf_3 = RandomForestClassifier(n_estimators=100, random_state=42)
fit_model(rf_clf_3, "Random Forest", X_train_pca4, y_train, X_test_pca4, y_test, dtype='With PCA4')

# Mesh grid for decision boundary visualization
h = 0.02  # Step size for mesh grid
x_min, x_max = X_pca_reduced[:, 0].min() - 1, X_pca_reduced[:, 0].max() + 1
y_min, y_max = X_pca_reduced[:, 1].min() - 1, X_pca_reduced[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh grid
# SVM Classifier (RBF)
svm_clf = SVC(kernel="rbf", C=0.34)  # Radial Basis Function (RBF) kernel with C=0.34
fit_model(svm_clf, "SVM (RBF)", X_train, y_train, X_test, y_test, dtype='Without PCA')

svm_clf_pca = SVC(kernel="rbf", C=0.55)  # Radial Basis Function (RBF) kernel with C=0.55
fit_model(svm_clf_pca, "SVM (RBF)", X_train_pca, y_train, X_test_pca, y_test, dtype='With PCA')

svm_clf_pca4 = SVC(kernel="rbf", C=0.55)  # Radial Basis Function (RBF) kernel with C=0.5
fit_model(svm_clf_pca4, "SVM (RBF)", X_train_pca4, y_train, X_test_pca4, y_test, dtype='With PCA4')

# Predict on mesh grid
# SVM Classifier (Linear)
svm_clf = SVC(kernel="linear", C=0.25)  # Linear kernel with C=0.25
fit_model(svm_clf, "SVM (Linear)", X_train, y_train, X_test, y_test, dtype='Without PCA')

svm_clf_pca = SVC(kernel="linear", C=0.25)  # Linear kernel with C=0.25
fit_model(svm_clf_pca, "SVM (Linear)", X_train_pca, y_train, X_test_pca, y_test, dtype='With PCA')

svm_clf_pca4 = SVC(kernel="linear", C=0.25)
fit_model(svm_clf_pca4, "SVM (Linear)", X_train_pca4, y_train, X_test_pca4, y_test, dtype='With PCA4')

# Convert the data into a pandas DataFrame
result_df = pd.DataFrame({
    "Method": list(model_results.keys()),
    "Accuracy(Without PCA)": [model_results[m]["Without PCA"]["Accuracy"] for m in model_results],
    "Accuracy(With PCA, n_comp=2)": [model_results[m]["With PCA"]["Accuracy"] for m in model_results],
    "Accuracy(With PCA, n_comp=4)": [model_results[m]["With PCA4"]["Accuracy"] for m in model_results]
})

# Sort by 'Without PCA Accuracy' in descending order
result_df = result_df.sort_values(by="Accuracy(Without PCA)", ascending=False)

# Display the table
display(result_df)

Z = svm_clf_pca4.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros((xx.ravel().shape[0], 2))])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

# Scatter plot of data points
sns.scatterplot(x=X_pca_reduced[:, 0], y=X_pca_reduced[:, 1], hue=df['Class'], palette=["blue", "red"], alpha=0.8)

# Labels and legend
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("SVM (Linear) Decision Boundary (Using PCA1 & PCA2)")
plt.legend()
plt.show()

# Convert the data into a pandas DataFrame
comparison_df = pd.DataFrame({
    "Method": ['Best Supervised (SVM (Linear))', 'Best Unsupervised (Kernel KMeans)'],
    "Accuracy": [model_results['SVM (Linear)']["Without PCA"]["Accuracy"], "{:.3f}".format(kernel_kmeans_accuracy)]
})

# Display the table
display(comparison_df)
