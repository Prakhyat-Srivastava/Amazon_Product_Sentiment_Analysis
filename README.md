# Amazon Product Sentiment Analysis 📊🔍

This repository contains a **machine learning project** focused on performing **sentiment analysis** on Amazon product reviews. The project uses **Logistic Regression** to classify reviews as either **Positive** or **Negative**. Special attention is given to handling **class imbalance** and fine-tuning the model to ensure a balanced performance between precision and recall for the minority class (negative feedback). The trained model and TF-IDF vectorizer are exported for future use.

## 📁 Dataset

The dataset used in this project consists of **Amazon Alexa reviews**. You can find and download it from Kaggle:

[Amazon Alexa Reviews - Kaggle Dataset](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews/data)

## 🚀 Project Overview

This project walks through the entire process of building a sentiment analysis model, starting with data preprocessing and ending with model export. The key steps include:

1. **Data Preprocessing**:
   - **Text Cleaning**: Removing stop words, punctuation, converting text to lowercase, and applying stemming.
   - **Feature Extraction**: Using **TF-IDF Vectorizer** to convert reviews into numerical form.
   
2. **Handling Class Imbalance**:
   - Tried **SMOTE** but ultimately used **Logistic Regression with class weights** to improve recall for the minority class (negative feedback).

3. **Threshold Adjustment**:
   - Fine-tuned the model's performance by adjusting the classification threshold based on **Precision-Recall curves**, aiming to balance precision and recall for both classes.

4. **Model Export**:
   - Exported the trained **Logistic Regression model** and **TF-IDF vectorizer** for future predictions using Pickle.

## 🏆 Final Model Performance

After threshold adjustment, the **Logistic Regression model** achieved the following performance:

- **Overall Accuracy**: 91.85%
- **Class 0 (Negative Feedback)**:
  - **Precision**: 58%
  - **Recall**: 62%
  - **F1-Score**: 60%

The final model provides a balanced detection of both positive and negative reviews, with an improved focus on correctly identifying negative feedback.

## 📚 Libraries and Dependencies

This project requires the following libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **scikit-learn**: For machine learning model building and evaluation.
- **nltk**: For natural language processing tasks such as text cleaning.
- **matplotlib**: For visualizing data during exploratory data analysis.
- **imblearn**: For handling class imbalance (SMOTE).
- **pickle**: For saving and loading the trained model and vectorizer.

To install all required libraries, run:
```bash
pip install pandas numpy scikit-learn nltk matplotlib imblearn
