# Fake News Detection Project

## Overview
This project focuses on addressing the critical issue of fake news detection using advanced data science and machine learning techniques. The solution leverages undersampling methods and fine-tuned LSTM models to achieve high accuracy in identifying fake news.

## Key Features
- **Undersampling Methods:**
  - Applied NearMiss undersampling technique on the Kaggle Fake and Real News Dataset.
  - Improved model accuracy by 10%, significantly enhancing the detection of misinformation.
- **Fake News Classifier:**
  - Developed a robust fake news classifier using LSTM (Long Short-Term Memory) models.
  - Utilized the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer for feature extraction.
  - Achieved an impressive 98% accuracy in detecting fake news.

## Dataset
- **Source:** Kaggle Fake and Real News Dataset
- **Content:** 44,898 news articles labeled as "FAKE" or "REAL".
- **Columns:** Title, Text (body text), Subject, and Date (publish date).

## Approach
### Data Preprocessing
- Removed stopwords, punctuation, and special characters.
- Employed TF-IDF vectorizer to convert textual data into numerical features.

### Addressing Data Imbalance
- **Technique:** Applied NearMiss undersampling to balance the dataset.
- **Effectiveness:** Evaluated the impact of undersampling by comparing model performance on balanced and imbalanced datasets.

### Model Selection and Training
- **Basic Models:** Naive Bayes, Logistic Regression, Support Vector Machine (SVM).
- **Advanced Models:** Decision Trees, Random Forest, LSTM networks.
- **Hyperparameter Tuning:** Used grid search and random search methods to find optimal hyperparameters.

### Evaluation Metrics
- Accuracy, Precision, Recall, and F1 Score.
- Emphasized Recall to ensure fake news is correctly identified.

## Results
- **Accuracy:** Achieved a 98% accuracy with the fine-tuned LSTM model.
- **Improvement:** Observed a 10% increase in accuracy with NearMiss undersampling.

## Conclusion
This project demonstrates the effectiveness of using LSTM models and undersampling techniques in detecting fake news. By improving model accuracy and addressing data imbalance, the solution provides a robust tool for combating misinformation.

## Future Work
- **Explore Different Architectures:** Investigate the impact of other machine learning and deep learning models.
- **Multimodal Approach:** Incorporate image and video analysis alongside text analysis.
- **Transfer Learning:** Leverage pre-trained models to improve performance and reduce training times.
- **Explainability:** Integrate techniques like LIME (Local Interpretable Model-Agnostic Explanations) to understand model decision-making processes.
- **Real-time Application:** Develop real-time detection systems for social media platforms and news websites.

## Installation and Usage
### Prerequisites
Ensure you have the following dependencies installed:
```bash
pip install tensorflow nltk pandas scikit-learn
```

### Clone Repository
```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### Running the Model
1. Preprocess the data:
```python
python preprocess.py
```
2. Train the model:
```python
python train.py
```
3. Evaluate the model:
```python
python evaluate.py
```

### Testing on New Articles
To test the model on new news articles, run:
```python
python predict.py --text "Your news article here"
```


## Contact
For questions or collaborations, contact reshmi14@uw.edu
