# 📧 Spam Detection using Machine Learning

A Machine Learning project to classify SMS messages as **Spam** or **Ham (Not Spam)** using **TF-IDF Vectorizer** and **Naive Bayes Classifier**.

---

## 📌 Features

- Load labeled spam/ham dataset
- Text preprocessing (cleaning)
- Convert text into numerical vectors using TF-IDF
- Train Naive Bayes classifier
- Evaluate model using Accuracy, Precision, Recall, and F1-Score
- Save the trained model and vectorizer
- Predict new SMS messages using the saved model

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Regular Expressions (re)

---

## 📂 Project Structure

```
Spam_Detection/
│── spam.csv
│── train_model.py
│── predict.py
│── model.pkl
│── vectorizer.pkl
│── README.md
```

---

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Spam-Detection-ML.git
```

Move into the project folder:

```bash
cd Spam-Detection-ML
```

Install required libraries:

```bash
pip install pandas scikit-learn joblib
```

---

## ▶️ Train the Model

Run:

```bash
python train_model.py
```

This will:

- Load the dataset
- Clean the text
- Convert text using TF-IDF
- Train the Naive Bayes model
- Display Accuracy and Classification Report
- Save:
  - `model.pkl`
  - `vectorizer.pkl`

---

## ▶️ Predict Messages

Run:

```bash
python predict.py
```

Example:

```
Enter Message:
Congratulations! You have won ₹50000.

Prediction:
SPAM
```

```
Enter Message:
Let's meet at 5 PM.

Prediction:
HAM
```

---

## 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

---

## 📚 Machine Learning Workflow

```
Dataset
   │
   ▼
Text Cleaning
   │
   ▼
TF-IDF Vectorizer
   │
   ▼
Naive Bayes Classifier
   │
   ▼
Model Evaluation
   │
   ▼
Save Model & Vectorizer
   │
   ▼
Prediction
```

---

## 🚀 Future Improvements

- GUI using Tkinter
- Flask Web Application
- Streamlit Deployment
- Logistic Regression Comparison
- Larger SMS Spam Dataset

---

## 👨‍💻 Author

**Karan Singh**

Machine Learning Project
