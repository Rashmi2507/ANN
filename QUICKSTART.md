# Quick Start Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ANN-Churn-Prediction.git
cd ANN-Churn-Prediction
```

### 2. Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

## Project Files Overview

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit web application |
| `experiments.ipynb` | Model training and experimentation |
| `prediction.ipynb` | Example predictions and testing |
| `model.h5` | Trained neural network model |
| `*.pkl` | Saved encoders and scaler |
| `Churn_Modelling.csv` | Training dataset |

## Model Training

To retrain the model with new data:

1. Open `experiments.ipynb` in Jupyter
2. Update the dataset path if needed
3. Run all cells sequentially
4. The model, encoders, and scaler will be saved automatically

## Troubleshooting

### Model not loading
- Ensure `model.h5` and `.pkl` files are in the project root
- Check that paths are correct

### Import errors
- Reinstall requirements: `pip install -r requirements.txt --upgrade`
- Restart your Python kernel/terminal

### Streamlit port already in use
```bash
streamlit run app.py --logger.level=debug --server.port 8502
```

## Expected Output

The app provides:
- ✅ Probability of customer churn
- 📊 Visual prediction confidence
- 💡 Model accuracy metrics

## Contact & Support

For issues or questions, please open an issue on GitHub.
