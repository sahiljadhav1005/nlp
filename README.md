# Fraud Detection in Transactions using NLP

## Course: NLP (Semester 6) - Pillai College of Engineering

## Team Members
- A 609 Nirmiti Borole  
- A 605 Faeek Baig  
- A 618 Sahil Jadhav  

## Acknowledgements
We would like to express our sincere gratitude to the following faculty members for their guidance and support:

**Subject Incharge:**  
- Dr. Sharvari Govilkar  

**Lab Incharge:**  
- Prof. Neha Ashok

---

## Project Overview
This mini project applies Natural Language Processing (NLP) and Machine Learning (ML) techniques to detect fraudulent financial transactions based on their textual descriptions. It leverages traditional ML models, deep learning architectures, and state-of-the-art transformer embeddings to achieve high accuracy in fraud classification.

---

## Dataset Details
- **Name:** Synthetic Financial Transaction Dataset  
- **Total Entries:** 1,000  
- **Columns:**
  - `Transaction_Description`: Text of the transaction
  - `Label`: "Fraudulent" or "Non-Fraudulent"

---

## Implementation Pipeline

### Section 1: Load the Dataset
- Dataset loaded using pandas.
- FastText model installed using pip and loaded for embedding generation.

### Section 2: Exploratory Data Analysis (EDA)
- Class distribution visualized using seaborn.
- Checked for null values, types, and general structure.

### Section 3: Text Preprocessing
- Lowercasing, special character removal.
- Stopword removal using NLTK.
- New column `cleaned_text` generated.

### Section 4: NLP-Based Feature Engineering
- Extracted character count, word count, average word length, and stopword count.

### Section 5: Vectorization Techniques
- Bag of Words (BoW)
- TF-IDF
- FastText embeddings (300-dim vectors using `cc.en.300.bin`)

### Section 6: Model Training and Evaluation
- Models trained using combined feature vectors.
- Metrics computed: Accuracy, Precision, Recall, and F1-score.

---

## Model Performance

| Model           | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|----------------|----------|-------------------|----------------|------------------|
| **Random Forest**     | 1.00     | 1.00              | 1.00           | 1.00             |
| **Logistic Regression** | 0.81     | 0.81              | 0.81           | 0.81             |
| **SVM**               | 0.69     | 0.71              | 0.69           | 0.68             |
| **CNN**               | 1.00     | 1.00              | 1.00           | 1.00             |
| **LSTM**              | 0.575    | 0.59              | 0.57           | 0.55             |
| **CNN-BiLSTM**        | 1.00     | 1.00              | 1.00           | 1.00             |

> All scores are calculated based on 200 test samples (100 per class).

---

## Key Findings
- CNN and CNN-BiLSTM models demonstrated perfect accuracy on the test set.
- Logistic Regression performed well with lower computational cost.
- SVM showed moderate accuracy with signs of class imbalance issues.
- FastText significantly improved feature richness and model performance.

---

## Future Scope
- Implement real-time detection pipeline.
- Incorporate user behavior metadata and temporal features.
- Apply attention-based models like BERT and RoBERTa.
- Introduce model interpretability for financial regulators.

---

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, NLTK, Seaborn, Matplotlib)
- **FastText** (Facebook Research)
- **TensorFlow / Keras** (for deep learning models)

---

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
