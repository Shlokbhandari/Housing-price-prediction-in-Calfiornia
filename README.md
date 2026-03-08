# California Housing Price Prediction 🏠📊

## Project Overview

This project builds a **Machine Learning model to predict housing prices in California** using the California Housing dataset.
The workflow covers the complete ML pipeline including **data preprocessing, feature transformation, model training, and prediction generation**.

The project demonstrates how to build an **end-to-end machine learning pipeline using Python and Scikit-Learn**.

---

## Dataset

The dataset used in this project is the **California Housing dataset**, which contains information about housing districts in California.

Some important features include:

* Median income
* Housing median age
* Total rooms
* Total bedrooms
* Population
* Households
* Latitude and longitude
* Ocean proximity (categorical feature)

Target variable:

```
median_house_value
```

---

## Project Workflow

The project follows a structured ML workflow:

1. **Data Loading**
2. **Stratified Train-Test Split**
3. **Data Preprocessing**
4. **Feature Scaling**
5. **Categorical Encoding**
6. **Pipeline Creation**
7. **Model Training**
8. **Model Saving**
9. **Prediction Generation**

---

## Technologies Used

* Python 
* Pandas
* NumPy
* Scikit-Learn
* Joblib

---

## Machine Learning Model

The model used in this project:

```
RandomForestRegressor
```

It is trained after applying a preprocessing pipeline that includes:

* Median imputation for missing values
* Standard scaling for numerical features
* One-hot encoding for categorical features

---

## Project Structure

```
.
├── housing.csv
├── main.py
├── model.pkl
├── pipeline.pkl
├── input.csv
├── output.csv
└── notebooks/
```

Notebooks include the step-by-step development of the ML workflow:

* Data exploration
* Train-test split
* Stratified sampling
* Data visualization
* Feature preprocessing
* Feature scaling
* Pipeline building

---

## How to Run the Project

### 1. Clone the repository

```
git clone https://github.com/Shlokbhandari/Housing-price-prediction-in-Calfiornia.git
```

### 2. Navigate to project folder

```
cd Housing-price-prediction-in-Calfiornia
```

### 3. Install dependencies

```
pip install pandas numpy scikit-learn joblib
```

### 4. Run the model

```
python main.py
```

---

## Output

After running the script:

* The trained model is saved as

```
model.pkl
```

* The preprocessing pipeline is saved as

```
pipeline.pkl
```

* Predictions are saved in

```
output.csv
```

---

## Learning Outcomes

Through this project, the following concepts were implemented:

* Data preprocessing pipelines
* Feature engineering
* Stratified sampling
* Machine learning model training
* Model serialization using Joblib
* End-to-end ML workflow implementation

---

## Author

**Shlok Bhandari**

B.E. Artificial Intelligence & Data Science
Machine Learning and Data Science Enthusiast 🚀
