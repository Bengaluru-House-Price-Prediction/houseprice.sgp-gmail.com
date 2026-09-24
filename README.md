# Bengaluru House Price Prediction

An AI/ML-based web application that predicts Bengaluru house prices using Machine Learning, Artificial Neural Network (ANN), and Flask.

## Project Overview

This project predicts house prices based on property-related features such as total square feet, number of bathrooms, balcony, BHK, and location.

The main objective of this project is to build an end-to-end AI/ML solution that includes data preprocessing, exploratory data analysis, feature engineering, model training, model evaluation, and web application development using Flask.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Flask
- HTML
- CSS
- Joblib

## Machine Learning

The project includes the following steps:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Scaling
- Machine Learning Model
- Artificial Neural Network (ANN)
- Model Evaluation

## Web Application

The trained model is integrated with a Flask web application.

The user enters the required property details, and the application predicts the estimated house price.

## Project Structure

```text
Bengaluru-House-Price-Prediction/
│
├── app.py
├── requirements.txt
│
├── models/
│   ├── ann_model.keras
│   ├── feature_columns.pkl
│   └── scaler.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```
## Model

The project uses Machine Learning and Artificial Neural Network (ANN) models for house price prediction.

The trained model takes property details as input and predicts the estimated house price.

## Input Features

The web application accepts the following property details:

- Total Square Feet
- Number of Bathrooms
- Number of Balconies
- Number of BHK
- Location

## Output

The application displays the predicted house price based on the property details entered by the user.

## How to Run

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required dependencies:

    pip install -r requirements.txt

4. Run the Flask application:

    python app.py

5. Open the local Flask URL in your web browser.

## Project Features

- House price prediction
- Machine Learning based prediction
- Artificial Neural Network (ANN)
- Feature scaling
- Input-based prediction
- Flask web application
- User-friendly web interface

## Future Improvements

- Improve model accuracy with more data.
- Add more property features.
- Deploy the application online.
- Improve the user interface.
- Add interactive visualizations.
- Implement additional machine learning models for comparison.

## Conclusion

This project demonstrates an end-to-end AI/ML-based house price prediction system.

It combines data preprocessing, exploratory data analysis, feature engineering, machine learning, artificial neural networks, and Flask web application development to provide a house price prediction system.

## Author

Bengaluru House Price Prediction Project
