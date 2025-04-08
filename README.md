# Laptop Price Prediction

## Project Overview

This project aims to predict the price of laptops based on various features such as brand, type, RAM, processor, and other key attributes. The model uses machine learning techniques to provide an estimated price, which can be useful for potential buyers, sellers, or anyone interested in understanding the price range for laptops based on their specifications.

The model utilizes a **stacked regression model**, combining the strengths of multiple machine learning algorithms to provide accurate and reliable price predictions. This is integrated into a user-friendly web application using **Streamlit**, allowing for easy interaction and real-time predictions.

## Table of Contents
- [Model Explanation](#model-explanation)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Files and Directories](#files-and-directories)
- [License](#license)

## Model Explanation

The laptop price prediction model is built using the following base algorithms:
1. **Random Forest Regressor**: A robust model that creates multiple decision trees and averages the results to prevent overfitting and improve predictions.
2. **Ridge Regression**: A linear model used to address multicollinearity, providing more stable predictions.
3. **XGBoost Regressor**: A powerful gradient boosting model that builds strong predictive models using an ensemble of weak learners.

These base models are then combined into a **Stacked Regressor**, where the predictions of each base model are used as inputs for a final regressor to generate the final output. The stacked model helps improve the overall prediction accuracy by leveraging the strengths of different models.

## Installation

To run this project on your local machine, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/laptop-price-predictor.git
    ```

2. **Navigate into the project directory**:
    ```bash
    cd laptop-price-predictor
    ```

3. **Install the required dependencies**:
    The easiest way to install the dependencies is by using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:
    After installing the dependencies, you can start the app using:
    ```bash
    streamlit run app.py
    ```

## Usage

Once the app is running, you can input the following details into the provided interface:
- **Brand**: The manufacturer of the laptop.
- **Type**: The type of laptop (e.g., gaming, ultrabook, etc.).
- **RAM (in GB)**: The amount of RAM in the laptop.
- **Operating System**: The operating system installed on the laptop (e.g., Windows, MacOS).
- **Weight (in kg)**: The weight of the laptop.
- **Touchscreen**: Whether the laptop has a touchscreen or not.
- **IPS Display**: Whether the laptop has an IPS display or not.
- **Screen Size (in inches)**: The size of the laptop’s screen.
- **Screen Resolution**: The screen resolution (e.g., 1920x1080).
- **CPU**: The CPU model used in the laptop.
- **HDD (in GB)**: The storage capacity for the hard disk drive (HDD).
- **SSD (in GB)**: The storage capacity for the solid-state drive (SSD).
- **GPU**: The brand of the laptop’s graphics processing unit (GPU).

After entering the values, the model will output an estimated price with a confidence interval, helping the user to understand the expected price range.

## Features

- **Laptop Price Prediction**: Based on input attributes, the model provides an estimated price for the laptop.
- **SHAP Visualizations**: Visualize the impact of each feature on the predicted price with SHAP (SHapley Additive exPlanations).
- **Interactive Web Application**: The Streamlit app allows users to interact with the model and get real-time predictions.

## Files and Directories

- **app.py**: The Streamlit application that hosts the web interface for users.
- **Laptop Price Predictor.ipynb**: The Jupyter Notebook containing data exploration, model building, and training.
- **traineddata.csv**: The dataset used to train the model.
- **requirements.txt**: A list of required Python libraries for the project.
- **pipeline.joblib**: The serialized machine learning model pipeline (preprocessing + model).
- **preprocessor.joblib**: The serialized preprocessing pipeline.
- **final_stacked_model.joblib**: The serialized final stacked model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by forking the repository, making your improvements, and submitting a pull request. Thank you for your interest in the Laptop Price Prediction Model!
