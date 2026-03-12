# QML-using-Quantum-Kernel-Matrix

## Project Overview

This project demonstrates Quantum Machine Learning (QML) using the Quantum Kernel Matrix approach. It leverages IBM quantum hardware to compute a kernel matrix encoding quantum similarities between data samples, which is then used to train a classical Support Vector Machine (SVM) for classifying breast cancer tumours as benign (B) or malignant (M).

## Workflow

1. **Data Preprocessing** (`processing code.py`)
   - Loads the raw Wisconsin Breast Cancer dataset (`data.csv`).
   - Applies standard scaling and PCA for dimensionality reduction.
   - Saves the reduced features and labels to `preprocessed data.csv`.

2. **Quantum Kernel Computation** (`quantum_kernel_matrix.py`)
   - Randomly samples 50 data points (`random_state=42`) from the preprocessed data.
   - Constructs parameterised quantum circuits (ZZFeatureMap) and runs them on IBM quantum hardware via Qiskit Runtime.
   - Computes the 50×50 kernel matrix and saves it to `quantum_kernel_matrix.csv`.

3. **SVM Training & Evaluation** (`training_svm.ipynb`)
   - Loads the kernel matrix and the **same** random sample of 50 data points used during kernel computation.
   - Normalises the kernel matrix to correct for quantum hardware noise: $K'_{ij} = K_{ij} / \sqrt{K_{ii} \cdot K_{jj}}$.
   - Trains an SVM with the precomputed kernel on an 80/20 stratified train-test split.
   - Evaluates accuracy on a single split and across 20 random splits with different seeds.
   - Visualises support vectors and class distribution using PC1 vs PC2.
   - Saves the trained model to `svm_model.joblib`.

## Project Structure

```
├── data.csv                     # Raw breast cancer dataset
├── preprocessed data.csv        # PCA-reduced features + labels
├── processing code.py           # Data preprocessing script
├── quantum_kernel_matrix.py     # Quantum kernel computation (IBM hardware)
├── quantum_kernel_matrix.csv    # Precomputed 50×50 quantum kernel matrix
├── training_svm.ipynb           # SVM training, evaluation & visualisation notebook
├── svm_model.joblib             # Saved trained SVM model
├── svm_support_vectors.png      # Support vector visualisation
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
├── LICENSE                      # Project licence
└── README.md                    # This file
```

## How to Run

### Prerequisites

- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Using the Jupyter Notebook

1. Ensure `preprocessed data.csv` and `quantum_kernel_matrix.csv` are in the project directory.
2. Open `training_svm.ipynb` in Jupyter or VS Code.
3. Run all cells top-to-bottom to:
   - Load and normalise the quantum kernel matrix
   - Train the SVM on a single split and report accuracy
   - Visualise support vectors (PC1 vs PC2)
   - Evaluate accuracy across 20 random train-test splits

### Running with Docker

1. Build the image:
   ```bash
   docker build -t qml-kernel .
   ```
2. Start the container:
   ```bash
   docker run -p 8888:8888 -v $(pwd):/app qml-kernel
   ```
   Or use Docker Compose:
   ```bash
   docker-compose up
   ```
3. Open `http://localhost:8888` in your browser and run `training_svm.ipynb`.

## Key Technical Details

- **Kernel normalisation**: The raw quantum kernel has diagonal values of ~0.73–0.80 due to gate noise and decoherence. Normalisation rescales the matrix so that $K(x_i, x_i) = 1$, improving SVM performance.
- **Data sampling alignment**: The notebook uses `.sample(n=50, random_state=42)` — matching the sampling in `quantum_kernel_matrix.py` — to ensure labels are correctly paired with kernel entries.
- **Accuracy resolution**: With 50 samples and an 80/20 split, there are only 10 test samples per split, so accuracy is quantised in 10% increments.
- **Multi-split evaluation**: 20 random splits (different seeds) are run to assess model stability. The loop uses scoped variable names to avoid overwriting the main train/test split state.

## Dependencies

| Package | Min Version |
|---------|-------------|
| pandas | 1.3.0 |
| numpy | 1.21.0 |
| scikit-learn | 1.0.0 |
| matplotlib | 3.4.0 |
| qiskit | 0.43.0 |
| qiskit-ibm-runtime | 0.15.0 |
| seaborn | 0.12.0 |
| joblib | 1.1.0 |
| tqdm | 4.62.0 |

## Notes

- The quantum kernel is precomputed, so the SVM cannot visualise the true decision boundary in the original feature space.
- To recompute the kernel on different hardware or with more samples, modify and run `quantum_kernel_matrix.py`.
- For data preprocessing changes, edit `processing code.py` and regenerate `preprocessed data.csv`.
