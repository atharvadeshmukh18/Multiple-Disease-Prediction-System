# Multiple Disease Prediction System

A comprehensive web-based application built with Streamlit that leverages machine learning models to predict the risk of multiple diseases including diabetes, heart disease, and Parkinson's disease. This system provides an intuitive interface for healthcare professionals and individuals to assess their disease risk based on clinical parameters.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Disease Prediction Models](#disease-prediction-models)
- [Dataset Information](#dataset-information)
- [Model Training](#model-training)
- [Input Parameters](#input-parameters)
- [Results Interpretation](#results-interpretation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

The Multiple Disease Prediction System is an intelligent health assessment tool that uses machine learning algorithms trained on medical datasets to predict the likelihood of various diseases. The application provides real-time predictions with a user-friendly interface, making it accessible to both healthcare professionals and the general public.

**Note:** This system is designed for educational and informational purposes and should not replace professional medical diagnosis.

## ✨ Features

- **Multi-Disease Support**: Predictions for three major diseases
  - Diabetes Prediction
  - Heart Disease Prediction
  - Parkinson's Disease Prediction

- **User-Friendly Interface**: Intuitive Streamlit-based web interface with organized input forms

- **Real-Time Predictions**: Instant risk assessment based on user input

- **Responsive Design**: Works seamlessly on desktop and tablet devices

- **Pre-trained Models**: Ready-to-use machine learning models trained on real-world medical datasets

- **Easy Navigation**: Sidebar menu for quick disease selection

## 📁 Project Structure

```
multiple-disease-prediction-system/
├── app.py                           # Main Streamlit application
├── encap.py                         # Encapsulation and utility functions
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── index.html                       # Frontend styling (optional)
├── tempCodeRunnerFile.py            # Temporary test file
│
├── saved_models/                    # Pre-trained ML models
│   ├── diabetes_model.sav          # Trained diabetes prediction model
│   ├── heart_disease_model.sav     # Trained heart disease prediction model
│   └── parkinsons_model.sav        # Trained Parkinson's disease prediction model
│
├── dataset/                         # Training datasets
│   ├── diabetes.csv                # Diabetes dataset
│   ├── heart.csv                   # Heart disease dataset
│   └── parkinsons.csv              # Parkinson's disease dataset
│
└── colab_files_to_train_models/     # Model training notebooks
    └── Multiple disease prediction system - heart.ipynb
```

## 💻 Tech Stack

- **Frontend & Backend**: Streamlit
- **ML Framework**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Serialization**: Pickle
- **UI Components**: Streamlit-option-menu
- **Development**: Python 3.8+
- **Jupyter**: For model training and experimentation

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/multiple-disease-prediction-streamlit-app.git
   cd Multiple-Disease-Prediction-System
   ```

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

   The application will launch in your default web browser at `http://localhost:8501`

## 📖 Usage

### Starting the Application

1. Execute the Streamlit command:
   ```bash
   streamlit run app.py
   ```

2. The application will open in your browser with a sidebar menu

3. Select the disease prediction module from the sidebar:
   - Diabetes Prediction
   - Heart Disease Prediction
   - Parkinson's Disease Prediction

### Making a Prediction

1. **Select Disease Type**: Choose from the sidebar menu

2. **Enter Health Parameters**: Fill in the required medical values in the input fields

3. **Submit for Analysis**: Click the prediction button ("Diabetes Test Result", "Heart Disease Test Result", etc.)

4. **View Results**: The prediction result will appear below in a success box

### Example Workflow

- Select "Diabetes Prediction" from the menu
- Enter glucose level, blood pressure, BMI, age, etc.
- Click "Diabetes Test Result"
- View the prediction: "The person is diabetic" or "The person is not diabetic"

## 🏥 Disease Prediction Models

### 1. Diabetes Prediction Model
**Algorithm**: Support Vector Machine (SVM) or Logistic Regression
- **Training Data**: Pima Indians Diabetes Dataset
- **Accuracy**: ~78-82%
- **Input Features**: 8 parameters

### 2. Heart Disease Prediction Model
**Algorithm**: Random Forest or Logistic Regression
- **Training Data**: Cleveland Heart Disease Dataset
- **Accuracy**: ~85-88%
- **Input Features**: 13 parameters

### 3. Parkinson's Disease Prediction Model
**Algorithm**: Support Vector Machine (SVM)
- **Training Data**: Oxford Parkinson's Disease Detection Dataset
- **Accuracy**: ~90%+
- **Input Features**: Multiple vocal and speech features

## 📊 Dataset Information

### Diabetes Dataset
- **File**: `dataset/diabetes.csv`
- **Rows**: 768 records
- **Features**: 8 medical attributes
- **Target**: Binary (0: Non-diabetic, 1: Diabetic)

### Heart Disease Dataset
- **File**: `dataset/heart.csv`
- **Rows**: 303 records
- **Features**: 13 clinical parameters
- **Target**: Binary (0: No disease, 1: Disease present)

### Parkinson's Disease Dataset
- **File**: `dataset/parkinsons.csv`
- **Rows**: 195 records
- **Features**: 22 vocal and speech-related measurements
- **Target**: Binary (0: Healthy, 1: Parkinson's disease)

## 🤖 Model Training

### Using Jupyter Notebooks

Training notebooks are provided in the `colab_files_to_train_models/` directory:

```bash
jupyter notebook colab_files_to_train_models/
```

### Training Steps

1. **Load Dataset**: Import CSV files from the dataset folder
2. **Data Preprocessing**: Handle missing values, scaling, and feature engineering
3. **Model Selection**: Train multiple algorithms and compare performance
4. **Hyperparameter Tuning**: Optimize model parameters
5. **Model Evaluation**: Validate using cross-validation and test sets
6. **Model Serialization**: Save trained models using pickle

### Retraining Models

To retrain models with new data:

1. Update the respective CSV file in `dataset/`
2. Run the training notebook
3. The new model will overwrite the old `.sav` file in `saved_models/`
4. Restart the Streamlit app to load the new model

## 📝 Input Parameters

### Diabetes Prediction Inputs
| Parameter | Unit | Range |
|-----------|------|-------|
| Pregnancies | count | 0-17 |
| Glucose | mg/dl | 0-199 |
| Blood Pressure | mmHg | 0-122 |
| Skin Thickness | mm | 0-99 |
| Insulin | µU/ml | 0-846 |
| BMI | kg/m² | 0-67.1 |
| Diabetes Pedigree Function | - | 0.078-2.42 |
| Age | years | 21-81 |

### Heart Disease Prediction Inputs
| Parameter | Description |
|-----------|-------------|
| Age | Patient age in years |
| Sex | 0: Female, 1: Male |
| Chest Pain Types | 1-4 (anginal to asymptomatic) |
| Resting Blood Pressure | mmHg |
| Serum Cholesterol | mg/dl |
| Fasting Blood Sugar | 0: ≤120 mg/dl, 1: >120 mg/dl |
| Resting ECG Results | 0-2 |
| Max Heart Rate | bpm |
| Exercise Induced Angina | 0: No, 1: Yes |
| ST Depression | Numeric value |
| Slope | Upslope/Flat/Downslope |
| Major Vessels | 0-4 (colored by fluoroscopy) |
| Thalassemia | 0: Normal, 1: Fixed, 2: Reversible |

### Parkinson's Disease Inputs
Multiple vocal and speech features including:
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
- MDVP:Flo(Hz) - Minimum vocal fundamental frequency
- Shimmer and Jitter measurements
- Harmonics-to-Noise Ratio (HNR)
- And additional speech-related parameters

## 📈 Results Interpretation

### Prediction Outputs

**Diabetes Prediction**:
- ✅ "The person is not diabetic" - Low risk
- ⚠️ "The person is diabetic" - High risk, consult physician

**Heart Disease Prediction**:
- ✅ "The person does not have any heart disease" - Low risk
- ⚠️ "The person is having heart disease" - High risk, seek medical attention

**Parkinson's Disease Prediction**:
- ✅ "Negative prediction" - No Parkinson's disease indicators
- ⚠️ "Positive prediction" - Possible Parkinson's disease, specialist consultation recommended

### Important Notes
- These predictions are probability-based estimates
- They should **never replace professional medical diagnosis**
- Always consult qualified healthcare professionals for definitive diagnosis
- Results are based on historical data patterns and may not apply to all individuals

## 🔧 Customization

### Adding a New Disease Prediction Module

1. **Train a Model**: Create a new Jupyter notebook and train your ML model
2. **Save the Model**: Pickle the trained model to `saved_models/`
3. **Update app.py**: Add new conditional block similar to existing disease predictions
4. **Add Menu Option**: Include the new disease in the sidebar menu

### Changing Model Algorithms

Retrain models using different algorithms:
- Random Forest
- Gradient Boosting
- Neural Networks (using TensorFlow/Keras)
- XGBoost
- AdaBoost

## 📦 Requirements

See `requirements.txt` for complete dependencies:
```
streamlit>=1.0
streamlit-option-menu>=0.2.1
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.2
```

Install all with:
```bash
pip install -r requirements.txt
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Models fail to load
- **Solution**: Ensure all `.sav` files exist in `saved_models/` directory

**Issue**: Import errors
- **Solution**: Run `pip install -r requirements.txt` and verify installation

**Issue**: Streamlit not starting
- **Solution**: Verify Python 3.8+ is installed, check virtual environment activation

**Issue**: NaN or unexpected predictions
- **Solution**: Verify input values are within expected ranges, check for missing fields

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn ML Algorithms](https://scikit-learn.org/)
- [Pandas Data Processing](https://pandas.pydata.org/)
- [Medical Dataset Repositories](https://archive.ics.uci.edu/ml/datasets.php)

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Areas
- Model improvements and retraining
- UI/UX enhancements
- New disease prediction modules
- Bug fixes and optimizations
- Documentation improvements

## ⚠️ Disclaimer

This application is designed for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Your Name/Organization**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset contributors and maintainers
- Streamlit team for the amazing framework
- Scikit-learn community for ML algorithms
- All contributors and testers

---

**Last Updated**: January 2026

**Status**: Active & Maintained ✅
