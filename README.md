# Customer Churn Prediction using Artificial Neural Networks

A machine learning project that predicts customer churn using an Artificial Neural Network (ANN) trained on bank customer data.

## 📋 Project Overview

This project builds and deploys an ANN model to predict whether a bank customer will churn (leave the bank). The model is trained on the Churn Modelling dataset and includes:
- Model training with TensorFlow/Keras
- Feature preprocessing and scaling
- Interactive Streamlit web application for predictions
- TensorBoard visualization of training metrics

## 🎯 Features

- **ANN Model**: Deep learning model with multiple hidden layers
- **Preprocessing**: Encoding categorical variables (Gender, Geography) and scaling numerical features
- **Web Interface**: Streamlit app for easy predictions
- **Model Artifacts**: Saved encoders and scalers for consistent preprocessing
- **Training Logs**: TensorBoard logs for monitoring training progress

## 📂 Project Structure

```
ANN/
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Model training notebook
├── prediction.ipynb                # Prediction example notebook
├── model.h5                        # Trained ANN model
├── scaler.pkl                      # StandardScaler for feature scaling
├── label_encoder_gender.pkl        # Label encoder for Gender
├── one_hot_encoder_geography.pkl   # One-hot encoder for Geography
├── Churn_Modelling.csv             # Training dataset
├── requirements.txt                # Python dependencies
├── logs/                           # TensorBoard logs
└── README.md                       # This file
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ANN.git
cd ANN
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Run the Streamlit App
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` where you can:
- Select customer demographics (Geography, Gender, Age)
- Enter financial information (Credit Score, Balance, Salary)
- Specify account details (Tenure, Number of Products, etc.)
- Get instant churn prediction

### View Training Notebooks

1. **experiments.ipynb** - Complete model training pipeline including:
   - Data loading and preprocessing
   - Categorical encoding
   - Feature scaling
   - ANN architecture
   - Model training with TensorBoard callbacks

2. **prediction.ipynb** - Example of making predictions on new data

## 📊 Model Details

- **Architecture**: 
  - Input Layer: 12 features
  - Hidden Layer 1: 64 neurons (ReLU activation)
  - Hidden Layer 2: 32 neurons (ReLU activation)
  - Output Layer: 1 neuron (Sigmoid activation)

- **Training**: 
  - Optimizer: Adam
  - Loss Function: Binary Crossentropy
  - Metrics: Accuracy
  - Early Stopping: Enabled for best weights

## 📈 Dataset

**Churn_Modelling.csv** contains 10,000 bank customer records with:
- Demographics: Age, Gender, Geography
- Financial: CreditScore, Balance, EstimatedSalary
- Account: Tenure, NumOfProducts, HasCrCard, IsActiveMember
- Target: Exited (1 = Churned, 0 = Retained)

## 🔧 Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Scikit-learn
- Pandas & NumPy
- Streamlit
- TensorBoard

See `requirements.txt` for full list.

## 📝 Usage Example

```python
# Load model and preprocessors
import tensorflow as tf
import pickle
import pandas as pd

model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input data with proper encoding
# ... (see prediction.ipynb for details)

# Make prediction
prediction = model.predict(scaled_input)
```

## 📊 Monitoring Training

View training metrics and validation loss with TensorBoard:
```bash
tensorboard --logdir logs/fit/
```

## 🎓 Key Learnings

- Feature preprocessing importance for neural networks
- Categorical encoding strategies (LabelEncoder, OneHotEncoder)
- Building and training neural networks with Keras
- Model deployment with Streamlit
- Handling imbalanced classification problems

## 📄 License

MIT License - feel free to use this project for learning or commercial purposes.

## 👤 Author

Created as a demonstration of end-to-end machine learning workflow.

## 🤝 Contributing

Feel free to fork, modify, and improve this project!

---

**Note**: The model and preprocessors are already trained and saved. To retrain the model, run the `experiments.ipynb` notebook.
