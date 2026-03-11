# QML-using-Quantum-Kernel-Matrix

## Project Overview
This project demonstrates Quantum Machine Learning (QML) using the Quantum Kernel Matrix approach. The workflow leverages quantum computing (IBM hardware) to compute a kernel matrix, which is then used to train a classical Support Vector Machine (SVM) for classifying breast cancer data as benign or malignant.

### Workflow
1. **Data Preparation**: The original dataset is preprocessed using Principal Component Analysis (PCA) to reduce dimensionality. The resulting features (PC1, PC2, etc.) and labels (benign/malignant) are saved in `preprocessed data.csv`.
2. **Quantum Kernel Calculation**: Quantum circuits are used to compute the kernel matrix, which captures quantum similarities between samples. The matrix is saved as `quantum_kernel_matrix.csv`.
3. **SVM Training**: The SVM is trained using the quantum kernel matrix and the labels from the preprocessed data. The model is evaluated for accuracy and visualized using the first two principal components.
4. **Evaluation**: The model's accuracy is tested on multiple train-test splits, and the support vectors are visualized.

## How It Works
- The quantum kernel matrix is computed using quantum circuits (see `quantum_kernel_matrix.py`).
- The SVM classifier uses this matrix as a precomputed kernel to learn the separation between benign and malignant cases.
- The workflow is implemented in both a Python script (`training svm.py`) and a Jupyter notebook (`training_svm.ipynb`).
- Visualization shows the support vectors and the distribution of classes in the reduced feature space.

## How to Run
1. **Install Dependencies**
	- Make sure you have Python 3.9+ installed.
	- Install all required packages:
	  ```bash
	  pip install -r requirements.txt
	  ```
2. **Prepare Data**
	- Ensure `preprocessed data.csv` and `quantum_kernel_matrix.csv` are present in the project directory.
3. **Run the SVM Training Script**
	- Execute the script to train and evaluate the SVM:
	  ```bash
	  python training svm.py
	  ```
	- The script will print the test accuracy and save the trained model as `svm_model.joblib`.
	- Visualization will be saved as `svm_support_vectors.png`.
4. **Use the Jupyter Notebook**
	- Open `training_svm.ipynb` in Jupyter or VS Code.
	- Run each cell to interactively train, evaluate, and visualize the SVM model.
	- The notebook includes running the model on multiple splits and visualizing results.

## Running with Docker

You can run the project in a containerized environment using Docker:

1. Build the Docker image:
   ```bash
   docker build -t qml-kernel .
   ```
2. Start the container:
   ```bash
   docker run -p 8888:8888 -v $(pwd):/app qml-kernel
   ```
   Or use docker-compose:
   ```bash
   docker-compose up
   ```
3. Access Jupyter Notebook:
   - Open your browser and go to `http://localhost:8888`.
   - Run `training_svm.ipynb` or any script interactively.

All dependencies and code will be available inside the container.

## Notes
- The project uses a precomputed quantum kernel, so the SVM cannot visualize the true hyperplane in 2D.
- For custom quantum kernel generation, see `quantum_kernel_matrix.py`.
- For further analysis or visualization, modify the notebook as needed.
