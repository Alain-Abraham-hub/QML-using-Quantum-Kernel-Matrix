
from django.shortcuts import render
import os
import joblib
import numpy as np

# Load the SVM model once at startup
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../svm_model.joblib'))
svm_model = joblib.load(MODEL_PATH)

def predict_view(request):
	prediction = None
	if request.method == 'POST':
		try:
			pc1 = float(request.POST.get('pc1'))
			pc2 = float(request.POST.get('pc2'))
			pc3 = float(request.POST.get('pc3'))
			pc4 = float(request.POST.get('pc4'))
			pc5 = float(request.POST.get('pc5'))
			features = np.array([[pc1, pc2, pc3, pc4, pc5]])
			# The model expects the same feature order as in training
			pred = svm_model.predict(features)[0]
			prediction = 'Malignant' if pred == 1 else 'Benign'
		except Exception as e:
			prediction = f'Error: {str(e)}'
	return render(request, 'predictor/form.html', {'prediction': prediction})
