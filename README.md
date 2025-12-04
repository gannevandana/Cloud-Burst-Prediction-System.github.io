# 🌩️ CloudBurst Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An advanced AI-powered web application for predicting cloudburst events using machine learning analysis of meteorological parameters.

## 📊 Model Performance

- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Accuracy**: 84.43%
- **F1-Score**: 83.26%
- **Training Dataset**: 145,460 weather records
- **Features**: 19 meteorological parameters
- **Cross-Validation**: 5-fold stratified

## ✨ Key Features

### 🤖 AI-Powered Predictions
- Utilizes state-of-the-art XGBoost algorithm
- Analyzes 19 different meteorological parameters
- Provides confidence scores and probability distributions

### 📊 Comprehensive Dashboard
- Real-time model performance metrics
- Algorithm comparison visualization
- Detailed statistical analysis

### 🎨 Beautiful UI
- Modern, responsive design with weather-themed colors
- Storm blues, cloud grays, and thunder purples color scheme
- Smooth animations and transitions
- Mobile-friendly interface

### 🔒 Robust Architecture
- RESTful API endpoints
- Input validation and error handling
- Data preprocessing pipeline
- Model persistence and caching

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Create project structure**:
```bash
mkdir cloudburst_flask_app
cd cloudburst_flask_app

# Create subdirectories
mkdir static static/css static/js static/images templates models
```

3. **Place all files in their respective directories**:
```
cloudburst_flask_app/
├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── results.html
│   ├── dashboard.html
│   └── about.html
└── models/
    ├── best_cloudburst_model.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    └── imputer.pkl
```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

5. **Copy your trained models**:
```bash
# Copy the generated model files to the models/ directory
cp best_cloudburst_model.pkl models/
cp scaler.pkl models/
cp label_encoder.pkl models/
cp imputer.pkl models/
```

6. **Run the application**:
```bash
python app.py
```

7. **Access the application**:
Open your browser and navigate to: `http://localhost:5000`

## 📁 Project Structure

### Backend (`app.py`)
- Flask web server configuration
- API endpoints for predictions
- Model loading and preprocessing
- Form data handling

### Frontend Templates
- **index.html**: Home page with features and statistics
- **predict.html**: Prediction form with 19 input parameters
- **results.html**: Display prediction results with visualizations
- **dashboard.html**: Model performance metrics and algorithm comparison
- **about.html**: Project information and technical details

### Styling (`style.css`)
- Custom CSS with weather-themed color scheme
- Responsive design for mobile and desktop
- Smooth animations and transitions
- Modern card-based layouts

### JavaScript (`script.js`)
- Form validation and error handling
- Real-time input validation
- Smooth scrolling and animations
- Mobile navigation toggle

## 🎯 How to Use

### Making a Prediction

1. Navigate to the **Predict** page
2. Fill in all required meteorological parameters:
   - **Temperature Data**: Min, Max, 9am, 3pm temperatures
   - **Precipitation**: Rainfall, evaporation, sunshine
   - **Wind Information**: Direction and speed measurements
   - **Humidity**: Morning and afternoon readings
   - **Pressure**: Atmospheric pressure values
   - **Cloud Cover**: Morning and afternoon observations

3. Click "Predict CloudBurst Tomorrow"
4. View results with:
   - Prediction (Yes/No)
   - Confidence percentage
   - Probability distribution
   - Safety recommendations

## 🔬 Input Parameters

| Parameter | Unit | Range | Required |
|-----------|------|-------|----------|
| Minimum Temperature | °C | -10 to 50 | Yes |
| Maximum Temperature | °C | -10 to 50 | Yes |
| Rainfall | mm | 0 to 500 | Yes |
| Evaporation | mm | 0 to 100 | No |
| Sunshine | hours | 0 to 24 | No |
| Wind Gust Direction | Compass | 16 directions | Yes |
| Wind Gust Speed | km/h | 0 to 200 | Yes |
| Wind Direction 9am | Compass | 16 directions | Yes |
| Wind Direction 3pm | Compass | 16 directions | Yes |
| Wind Speed 9am | km/h | 0 to 150 | Yes |
| Wind Speed 3pm | km/h | 0 to 150 | Yes |
| Humidity 9am | % | 0 to 100 | Yes |
| Humidity 3pm | % | 0 to 100 | Yes |
| Pressure 9am | hPa | 900 to 1100 | Yes |
| Pressure 3pm | hPa | 900 to 1100 | Yes |
| Cloud Cover 9am | oktas | 0 to 8 | No |
| Cloud Cover 3pm | oktas | 0 to 8 | No |
| Temperature 9am | °C | -10 to 50 | Yes |
| Temperature 3pm | °C | -10 to 50 | Yes |

## 🎨 Color Scheme

The application uses a weather-themed color palette:

- **Primary Color**: Deep Blue (#1e3a8a) - Representing storm clouds
- **Secondary Color**: Sky Blue (#3b82f6) - Clear sky elements
- **Accent Color**: Purple (#8b5cf6) - Thunder and lightning
- **Danger Color**: Red (#ef4444) - Warnings and alerts
- **Success Color**: Green (#10b981) - Safe conditions
- **Warning Color**: Orange (#f59e0b) - Caution states

## 🔧 API Endpoints

### POST `/predict`
Make a cloudburst prediction

**Request Body** (form data):
- All 19 meteorological parameters

**Response**: Renders results page with prediction

### POST `/api/predict`
JSON API endpoint for predictions

**Request Body** (JSON):
```json
{
  "MinimumTemperature": 13.4,
  "MaximumTemperature": 22.9,
  "Rainfall": 0.6,
  ...
}
```

**Response** (JSON):
```json
{
  "success": true,
  "prediction": "No",
  "confidence": 87.56,
  "probabilities": {
    "No": 87.56,
    "Yes": 12.44
  }
}
```

## 🤖 Machine Learning Pipeline

1. **Data Preprocessing**:
   - Missing value imputation (median strategy)
   - Feature encoding for categorical variables
   - Standard scaling for numerical features

2. **Model Training**:
   - Algorithm: XGBoost Classifier
   - Training samples: 116,368
   - Testing samples: 29,092
   - Cross-validation: 5-fold

3. **Evaluation Metrics**:
   - Accuracy: 84.43%
   - Precision: High precision for both classes
   - Recall: Balanced recall scores
   - F1-Score: 83.26%

## 📈 Algorithms Compared

During development, 9 algorithms were evaluated:

1. **XGBoost** ⭐ (Selected - Best Performance)
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors
6. Decision Tree
7. Logistic Regression
8. Naive Bayes
9. AdaBoost

## ⚠️ Important Notes

- This system provides predictions based on statistical analysis
- Should be used as supplementary tool with official forecasts
- Accuracy: 84.43% (margin of error exists)
- Always follow official weather warnings
- Designed for educational and research purposes

## 🛠️ Technologies Used

### Backend
- **Flask**: Web framework
- **Python**: Programming language
- **XGBoost**: Machine learning algorithm
- **Scikit-learn**: ML utilities and preprocessing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Joblib**: Model serialization

### Frontend
- **HTML5**: Structure and semantics
- **CSS3**: Styling and animations
- **JavaScript**: Interactivity and validation
- **Responsive Design**: Mobile-first approach

## 📊 Dataset Information

- **Total Records**: 145,460 weather observations
- **Features**: 19 meteorological parameters
- **Target Variable**: CloudBurstTomorrow (Binary: Yes/No)
- **Time Period**: Historical weather data
- **Locations**: Multiple geographical areas

## 🚀 Deployment

### Local Deployment
```bash
python app.py
# Access at http://localhost:5000
```

### Production Deployment

**Using Gunicorn**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Using Docker**:
```dockerfile
FROM python:3.8
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

## 🔒 Security Considerations

- Input validation on all form fields
- CSRF protection (configure SECRET_KEY)
- Rate limiting for API endpoints (recommended)
- HTTPS in production (recommended)

## 📝 License

MIT License - See LICENSE file for details

## 👨‍💻 Developer

Developed with ❤️ for Weather Safety and Research

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📞 Support

For questions or support, please open an issue on the project repository.

---

**⚠️ Disclaimer**: This application is for educational and research purposes. 
Always consult official meteorological services for critical weather decisions.

**🌩️ Stay Safe, Stay Informed!**
