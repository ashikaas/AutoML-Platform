# 🧠 AutoML Platform

An end-to-end **Automated Machine Learning (AutoML)** application built using **Python, Streamlit, and Scikit-Learn**. The platform automatically preprocesses datasets, detects the machine learning task, trains multiple models, compares their performance, visualizes results, and allows users to download the best trained model.

---

## 🚀 Live Demo

**Coming Soon**

---

## 💻 GitHub Repository

**https://github.com/ashikaas/AutoML-Platform**

---

## 📌 Features

- 📂 Upload any CSV dataset
- 🧹 Automatic data preprocessing
- 🎯 Target column selection
- 🤖 Automatic Classification / Regression detection
- ⚡ Train multiple Machine Learning models
- 📊 Compare model performance
- 📈 Correlation Heatmap
- 📉 Model Performance Visualization
- 📋 Confusion Matrix for Classification
- 💾 Download the best trained model (.pkl)
- 📄 Download model comparison results (.csv)
- 🎨 Interactive Streamlit interface

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib
- XGBoost

---

## 📂 Project Structure

```text
AutoML-Platform/
│
├── app.py
├── data_loader.py
├── preprocessing.py
├── model_trainer.py
├── evaluator.py
├── predictor.py
├── explainability.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── sample_dataset/
│   └── titanic.csv
│
└── screenshots/
    ├── 01_homepage.png
    ├── 02_dataset_overview.png
    ├── 03_correlation_heatmap.png
    ├── 04_model_comparison.png
    ├── 05_model_performance.png
    └── 06_confusion_matrix.png
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ashikaas/AutoML-Platform.git
```

### 2. Navigate to the Project Folder

```bash
cd AutoML-Platform
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📊 Workflow

1. Upload a CSV dataset.
2. Select the target column.
3. The application automatically:
   - Cleans the dataset
   - Preprocesses the data
   - Detects the machine learning task
   - Trains multiple machine learning models
   - Evaluates model performance
   - Selects the best-performing model
4. Visualize model performance.
5. Download the trained model and performance report.

---

# 📸 Screenshots

## 🏠 Home Page

![Homepage](screenshots/01_homepage.png)

---

## 📊 Dataset Overview

![Dataset Overview](screenshots/02_dataset_overview.png)

---

## 📈 Correlation Heatmap

![Correlation Heatmap](screenshots/03_correlation_heatmap.png)

---

## 📋 Model Comparison

![Model Comparison](screenshots/04_model_comparison.png)

---

## 📊 Model Performance

![Model Performance](screenshots/05_model_performance.png)

---

## 📉 Confusion Matrix

![Confusion Matrix](screenshots/06_confusion_matrix.png)

---

## 📁 Sample Dataset

A sample **Titanic dataset** is included inside the **sample_dataset** folder for quick testing.

---

## 🎯 Future Improvements

- Hyperparameter tuning
- Cross-validation support
- SHAP explainability
- Additional Machine Learning algorithms
- Automated feature selection
- Model deployment support
- Auto-generated ML reports
- Advanced data visualization dashboard

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing
- Machine Learning workflows
- Classification and Regression
- Model evaluation
- Data visualization
- Streamlit application development
- Python project structuring
- Model serialization using Joblib

---

## 👨‍💻 Author

**Ashika Srivastava**

B.Tech Computer Science Engineering (Artificial Intelligence)

MIT Art, Design & Technology University, Pune

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is licensed under the MIT License.
