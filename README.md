# Vertebral-Column-Classification-using-Kernel-K-Means-and-SVM

This project analyses the Vertebral Column dataset to classify patients as Normal (0) or Abnormal (1). It includes exploratory data analysis, dimensionality reduction using PCA, unsupervised clustering (Kernel K-Means, GMM, Hierarchical), and supervised models (Naive Bayes, Random Forest, SVM with Linear and RBF kernels). Performance is compared across models with and without PCA.

## Dataset
- File: `6156538.txt`  
- Features: 6 biomechanical attributes  
- Target: Class (NO → 0, AB → 1)

## Workflow

### 1. Exploratory Data Analysis
- Checked for missing or non-numeric values  
- Correlation heatmap  
- Violin plots, histograms, and pairplots to compare class distributions  

### 2. Dimensionality Reduction
- Standardized features  
- PCA to 2 and 4 components  
- Visualised PCA component contributions  

### 3. Unsupervised Learning
Evaluated:
- **Kernel K-Means (with Nystroem kernel approximation)**
- **Gaussian Mixture Model (GMM)**
- **Agglomerative Clustering**

Metrics:
- Adjusted Rand Index (ARI)  
- Cluster-to-class mapping accuracy  

Best unsupervised result:  
- **Kernel K-Means** (highest ARI and accuracy)

### 4. Supervised Learning
Models tested:
- Naive Bayes  
- Random Forest  
- SVM (Linear and RBF kernels)

Evaluations done:
- With original features  
- With PCA (2 components)  
- With PCA (4 components)

Best supervised model:  
- **SVM (Linear Kernel)** without PCA  
  (highest accuracy among supervised methods)

### 5. Decision Boundary Visualisation
- Linear SVM decision boundary plotted using PCA-reduced space  

## Results Summary
- **Best unsupervised:** Kernel K-Means  
- **Best supervised:** SVM (Linear)  
- PCA improved visualisation but not always accuracy  

## How to Run
1. Place `6156538.txt` in the project folder  
2. Run the Jupyter notebook  
3. All plots, PCA outputs, and model comparison tables will generate automatically  

## Dependencies
- numpy, pandas  
- matplotlib, seaborn  
- scikit-learn  
- scipy  
