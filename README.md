# Email/SMS Spam Classifier

A machine learning project that classifies email or SMS messages as either **Spam** or **Not Spam** using a trained Naive Bayes model. The project employs Natural Language Processing (NLP) techniques for text preprocessing and integrates a user-friendly web interface built with Streamlit.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Model Training Pipeline](#model-training-pipeline)
6. [Web App](#web-app)
7. [Installation](#installation)
8. [How to Use](#how-to-use)
9. [Results](#results)
10. [Future Enhancements](#future-enhancements)

---

## Project Overview

The **Email/SMS Spam Classifier** analyzes text messages and predicts whether they are spam or not. It uses a pre-trained Multinomial Naive Bayes model and vectorization techniques like TF-IDF for feature extraction. This project demonstrates the application of text preprocessing, machine learning, and deployment using Streamlit.

---

## Features

- Preprocesses text messages using:
  - Tokenization
  - Stopword removal
  - Stemming
- Transforms text into numerical features using TF-IDF Vectorization.
- Classifies messages as "Spam" or "Not Spam" with high accuracy.
- Provides an interactive and simple web interface.

---

## Technologies Used

- **Python**
- **Streamlit** (Web application framework)
- **Scikit-learn** (Machine Learning)
- **NLTK** (Natural Language Toolkit)
- **Pandas** (Data manipulation)
- **NumPy** (Numerical computing)
- **Matplotlib** (Visualization for analysis)
- **Pickle** (Model persistence)

---

## Dataset

The dataset used for training and testing the model is publicly available and consists of labeled SMS messages. Each message is categorized as either "spam" or "ham" (not spam). The dataset includes columns for the message content and its respective label.
Dataset link: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

Example:

| Label | Message                 |
|-------|-------------------------|
| ham   | Hi, how are you?        |
| spam  | Win $1000 now! Click... |

---

## Model Training Pipeline

1. **Data Preprocessing**:
   - Lowercasing all text.
   - Removing stopwords and punctuation.
   - Stemming words to their root forms.

2. **Vectorization**:
   - Applying TF-IDF (Term Frequency-Inverse Document Frequency) for numerical feature extraction from text.

3. **Model Selection**:
   - Using `MultinomialNB` (Naive Bayes classifier) for its suitability with text classification.

4. **Evaluation**:
   - Accuracy and precision scores are computed to evaluate model performance.

5. **Model Saving**:
   - Trained TF-IDF vectorizer and Naive Bayes model are saved as `vectorizer.pkl` and `model.pkl` respectively for use in the web application.

---

## Web App

The project includes a Streamlit-based web application for real-time spam classification. Users can input a text message, and the app will classify it as either "Spam" or "Not Spam."

---

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SubhamKumar-Gaurav/Email-SMS-Spam-Classifier.git
   cd Email-SMS-Spam-Classifier
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Resources** (if not already downloaded):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

5. **Run the Web Application**:
   ```bash
   streamlit run app.py
   ```

---

## How to Use

1. Navigate to the running Streamlit web app (usually at `http://localhost:8501`).
2. Enter an SMS or email message into the input field.
3. Click on the "Predict" button.
4. View the classification result: "Spam" or "Not Spam."

---

## Results

### Model Performance:
- **Accuracy**: Achieved high accuracy during testing (>95%).
- **Precision**: Ensures minimal false positives for spam messages.

---

## Future Enhancements

1. **Dataset Expansion**:
   - Include more diverse datasets for better generalization.
2. **Advanced Models**:
   - Experiment with advanced algorithms like XGBoost or transformers (e.g., BERT).
3. **Deployment**:
   - Deploy the web app on a cloud platform like AWS, Heroku, or Streamlit Cloud.
4. **Multilingual Support**:
   - Add support for classifying messages in different languages.

---

## Repository Structure

```
Email-SMS-Spam-Classifier/
├── app.py                # Streamlit web application
├── model.pkl             # Trained Naive Bayes model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── data/                 # Dataset (optional, if included)
```

---

## Contributing

Contributions are welcome! If you have suggestions for improvements or want to report an issue, feel free to create a pull request or an issue in this repository.

---


## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing SMS spam datasets.
- [NLTK Documentation](https://www.nltk.org/) for text preprocessing techniques.
- [Streamlit](https://streamlit.io/) for simplifying web app development.

---
  
## Contact
For any questions or feedback, please reach out to:

Subham Kumar Gaurav: subhamgaurav2001@gmail.com  
GitHub: [SubhamKumar-Gaurav](https://github.com/SubhamKumar-Gaurav)

---
