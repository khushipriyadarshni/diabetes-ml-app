# Diabetes Risk Prediction ML Project

A minimal, production-ready machine learning project for predicting diabetes risk based on patient features.

## Features

- **Training Pipeline**: Automated data preprocessing, model training, and evaluation
- **Interactive Web App**: Streamlit-based interface for single and batch predictions
- **Input Validation**: Comprehensive validation with clear error messages
- **Zero-as-Missing Handling**: Properly handles medically invalid zero values
- **Model Metrics**: Tracks and displays ROC-AUC, accuracy, precision, recall, and F1 score

## Project Structure

```
diabetes-ml-app/
├── data/
│   └── data/
│       └── diabetes.csv          # Training dataset
├── models/
│   ├── model.joblib              # Trained model (generated)
│   └── model_metrics.json        # Training metrics (generated)
├── train.py                      # Training script
├── utils.py                      # Utility functions
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to train the model:

```bash
python train.py
```

This will:
- Load data from `data/data/diabetes.csv`
- Perform exploratory data analysis
- Preprocess data (handle zero-as-missing values)
- Split data into train/validation sets (stratified)
- Train a RandomForestClassifier with StandardScaler
- Evaluate on validation set
- Save model to `models/model.joblib`
- Save metrics to `models/model_metrics.json`

### Running the Web App

Launch the Streamlit application:

```bash
streamlit run app.py
```

The app provides:
- **Single Prediction**: Input patient features via sidebar and get instant predictions
- **Probability Gauge**: Visual representation of diabetes risk probability
- **Threshold Slider**: Adjustable classification threshold (default: 0.5)
- **Batch Prediction**: Upload CSV file with patient data for bulk predictions
- **Model Metrics**: Display of model performance metrics
- **Feature Ranges**: Information about valid input ranges

## Input Features

The model uses 8 features:

1. **Pregnancies**: Number of pregnancies (0-20)
2. **Glucose**: Plasma glucose concentration (0-300, cannot be 0)
3. **BloodPressure**: Diastolic blood pressure (0-200, cannot be 0)
4. **SkinThickness**: Triceps skin fold thickness (0-100, cannot be 0)
5. **Insulin**: 2-Hour serum insulin (0-1000, cannot be 0)
6. **BMI**: Body mass index (0-70, cannot be 0)
7. **DiabetesPedigreeFunction**: Diabetes pedigree function (0.0-3.0)
8. **Age**: Age in years (0-120)

## Data Preprocessing

The training pipeline handles:
- **Zero-as-Missing**: For Glucose, BloodPressure, SkinThickness, Insulin, and BMI, zero values are treated as missing and imputed with median
- **Standardization**: Features are standardized using StandardScaler
- **Stratified Splitting**: Train/validation split maintains class distribution

## Model Architecture

- **Preprocessing**: StandardScaler
- **Classifier**: RandomForestClassifier (100 estimators, random_state=42)
- **Evaluation Metrics**: ROC-AUC, Accuracy, Precision, Recall, F1 Score

## Batch Prediction

For batch predictions, upload a CSV file with the same column names as the training data:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

The app will validate inputs and return predictions with probabilities and risk levels.

## Error Handling

The application includes comprehensive error handling for:
- Missing values
- Negative values
- Invalid zero values (for medically invalid columns)
- Out-of-range values
- Type conversion errors

All errors display clear, user-friendly messages.

## Dependencies

- pandas==2.2.3
- numpy==2.1.3
- scikit-learn==1.5.2
- joblib==1.4.2
- streamlit==1.39.0
- plotly==5.24.1

## License

This project is provided as-is for educational and demonstration purposes.

