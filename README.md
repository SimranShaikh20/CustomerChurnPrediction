# Customer Churn Prediction Using Artificial Neural Network

## Overview
This project focuses on predicting **customer churn** in the telecom business using deep learning techniques, specifically **Artificial Neural Networks (ANN)**. Customer churn prediction helps businesses understand why customers are leaving and take proactive measures to retain them.

The model evaluates churn using key performance metrics like **Precision**, **Recall**, and **F1-Score**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SimranShaikh20/CustometChurnPrediction/blob/main/CustomerChurnPrediction.ipynb)

---

## Table of Contents
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Dataset
The project uses a telecom industry dataset containing customer information, such as:
- Customer ID
- Demographics (e.g., gender, senior citizen status)
- Services (e.g., phone service, internet service)
- Account Information (e.g., tenure, charges)

### Target Variable:
- **Churn**: Binary variable indicating if a customer has left the service (Yes/No).

### Preprocessing Steps:
1. **Handling missing data**: Rows with spaces in `TotalCharges` are removed.
2. **Feature Engineering**: Conversion of categorical variables to numerical formats.
3. **Scaling**: Normalization or standardization of numerical features.

---

## Technologies Used
- **Python**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **TensorFlow/Keras**: Building and training the Artificial Neural Network

---

## Installation
To run this project locally:
1. Clone the repository:
   ```bash
   git clone https://github.com/SimranShaikh20/CustometChurnPrediction.git
   cd CustometChurnPrediction
   ```
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow
   ```
3. Run the notebook:
   - Open Jupyter Notebook or Google Colab.
   - Upload and run the `CustomerChurnPrediction.ipynb` file.

---

## Project Workflow
1. **Data Exploration**:
   - Inspecting the dataset
   - Visualizing distributions of key features
2. **Data Preprocessing**:
   - Handling missing or invalid values
   - Encoding categorical features
   - Scaling numerical data
3. **Model Building**:
   - Creating a deep learning model using TensorFlow/Keras
   - Configuring input layers, hidden layers, and output layers
   - Applying activation functions
4. **Model Training**:
   - Splitting data into training and testing sets
   - Fitting the ANN model on training data
5. **Evaluation**:
   - Measuring Precision, Recall, and F1-Score
   - Visualizing the confusion matrix and learning curves

---

## Results
The model predicts customer churn with significant accuracy. Key evaluation metrics include:
- **Precision**: Measures positive predictive value
- **Recall**: Measures ability to identify churn cases
- **F1-Score**: Balances Precision and Recall

Visualization results include:
- Confusion Matrix
- Accuracy and Loss curves

---

## Contributing
Contributions are welcome! Follow these steps:
1. Fork this repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-branch
   ```
5. Open a pull request.

---

## License
This project is licensed under the **MIT License**.

---

## Contact
For questions or suggestions, feel free to reach out:
- **Name**: Simran
- **GitHub**: [SimranShaikh20](https://github.com/SimranShaikh20)
