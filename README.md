# 📊 Support Ticket Classification System (NLP)

## 🚀 Project Overview

This project builds a Machine Learning system to automatically classify customer support tickets into predefined categories using Natural Language Processing (NLP).

Support teams receive thousands of tickets daily. Manually sorting them is time-consuming and inefficient. This system automates the process, improving response time and operational efficiency.

---

## 🎯 Objective

* Classify support tickets based on text
* Automate ticket routing
* Reduce manual workload
* Improve customer support efficiency

---

## 📂 Dataset

* IT Support Ticket Dataset (~47,000 records)
* Columns used:

  * `Document` → Ticket Description
  * `Topic_group` → Ticket Category

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* NLTK (Text preprocessing)
* Scikit-learn (ML models)
* TF-IDF Vectorization
* Naive Bayes Classifier
* Matplotlib & Seaborn

---

## ⚙️ Project Workflow

1. **Data Loading**

   * Loaded large-scale ticket dataset

2. **Text Preprocessing**

   * Lowercasing
   * Removing punctuation & numbers
   * Stopword removal

3. **Feature Engineering**

   * Converted text into numerical features using TF-IDF

4. **Model Training**

   * Used Multinomial Naive Bayes

5. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix

---

## 📈 Model Performance

* **Accuracy:** ~77.5%

### Key Observations:

* Strong performance on major categories like Hardware and HR Support
* Some confusion between similar categories due to overlapping text

---

## 📊 Confusion Matrix

The confusion matrix visualizes model performance across all classes, showing correct predictions and misclassifications.
<img width="801" height="675" alt="image" src="https://github.com/user-attachments/assets/da07ffc9-c8f1-45ba-a5f8-4f2fa30b7fcc" />

---

## 💼 Business Impact

* Automates ticket categorization
* Reduces manual sorting effort
* Speeds up response time
* Improves support team productivity
* Enhances customer satisfaction

---

## 🧠 Key Insights

* Text data is highly effective for category classification
* Similar categories may overlap in wording
* Priority prediction requires additional business context

---

## 📁 Project Structure

FUTURE_ML_02-main/
│
├── data/
│   └── new_data.csv
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluation.py
│
├── outputs/
│   └── predictions.csv
│
├── main.py
├── requirements.txt
└── README.md

---

## ▶️ How to Run

1. Clone the repository:
   git clone https://github.com/your-username/FUTURE_ML_02-main.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the project:
   python main.py

---

## 📤 Output

Predictions are saved in:
outputs/predictions.csv

---

## 🔮 Future Improvements

* Try advanced models (SVM, Logistic Regression, BERT)
* Handle class imbalance
* Build API for real-time predictions
* Improve feature engineering

---

## 📌 Conclusion

This project demonstrates how NLP and Machine Learning can automate real-world support operations and improve efficiency in handling customer tickets.

---

## 📢 Author

**Kanishk**
Machine Learning Enthusiast

---

## ⭐ If you found this useful, consider giving it a star!
