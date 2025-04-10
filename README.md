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
This mini project applies Natural Language Processing (NLP) and Machine Learning (ML) techniques to detect fraudulent financial transactions based on their textual descriptions. We use traditional ML algorithms, deep learning architectures, and state-of-the-art transformer models such as BERT and RoBERTa to classify transactions as fraudulent or non-fraudulent.

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
- Loaded using pandas.
- FastText model installed via pip and used for embeddings.

### Section 2: Exploratory Data Analysis (EDA)
- Visualized label distribution.
- Checked for null values and basic structure.

### Section 3: Text Preprocessing
- Lowercasing, removing special characters.
- Tokenization, stopword removal using NLTK.
- New `cleaned_text` column generated.

### Section 4: NLP-Based Feature Engineering
- Char count, word count, average word length, stopword count.

### Section 5: Feature Generation
- **Bag of Words (BoW)**
- **TF-IDF**
- **FastText Embeddings (300-dim)**

### Section 6: Model Training

#### Traditional ML Models:
- Random Forest  
- Logistic Regression  
- Support Vector Machine (SVM)

#### Deep Learning Models:
- Convolutional Neural Network (CNN)  
- Long Short-Term Memory (LSTM)  
- CNN + BiLSTM  

#### Transformer Language Models:
- **BERT**: Fine-tuned `bert-base-uncased` model with added classification layer  
- **RoBERTa**: Fine-tuned `roberta-base` model for binary classification

---

## Model Performance

| Model           | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) |
|----------------|----------|-------------------|----------------|------------------|
| **Random Forest**     | 0.93     | 0.94              | 0.92           | 0.93             |
| **Logistic Regression** | 0.81     | 0.81              | 0.81           | 0.81             |
| **SVM**               | 0.69     | 0.71              | 0.69           | 0.68             |
| **CNN**               | 0.95     | 0.95              | 0.95           | 0.95             |
| **LSTM**              | 0.57     | 0.59              | 0.57           | 0.55             |
| **CNN-BiLSTM**        | 0.96     | 0.96              | 0.96           | 0.96             |
| **BERT**              | 0.96     | 0.96              | 0.96           | 0.96             |
| **RoBERTa**           | 0.98     | 0.98              | 0.98           | 0.98             |


---

## Key Findings
- **CNN, CNN-BiLSTM, BERT, and RoBERTa** performed excellently with minimal performance drop on validation sets.
- **LSTM** underperformed, suggesting architectural limitations or training constraints.
- **RoBERTa** yielded the best overall performance, making it the preferred model for production.
- **FastText embeddings** greatly enhanced feature representation quality.

---

## Future Scope
- Deploy real-time fraud detection API.
- Use behavioral and transactional metadata for deeper insights.
- Incorporate explainability methods (e.g., SHAP, LIME) for transparency.
- Expand detection to multilingual datasets using models like XLM-R.

---

## Technologies Used
- **Python** (Pandas, NumPy, Scikit-learn, NLTK, Seaborn, Matplotlib)
- **FastText** (Facebook Research)
- **TensorFlow / Keras** (DL Models)
- **Transformers (Hugging Face)** for BERT & RoBERTa

---


## Conclusion
This project demonstrates the effectiveness of combining NLP techniques with modern machine learning and deep learning models for fraud detection in transactional data. By analyzing the semantics and contextual patterns in transaction descriptions, the models can accurately distinguish between fraudulent and legitimate activities. The results show that transformer-based models like RoBERTa outperform traditional models, indicating their strong capability in understanding natural language nuances. Going forward, the integration of more real-world data, explainability techniques, and real-time systems will further enhance the system’s utility in financial fraud prevention.

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training pipeline:
   ```bash
   python fraud_detection_pipeline.py
   ```
4. For Transformer models, open:
   ```bash
   jupyter notebook bert_roberta_fraud_detection.ipynb
   ```

---
