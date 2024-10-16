
# AI-Based Phishing Email Classifier

This tool classifies emails as either phishing or legitimate using machine learning and Natural Language Processing (NLP). The classifier is trained on email content and can detect phishing attempts based on textual patterns.

## Features
- Detects phishing emails based on the content (subject, body).
- Trained using TF-IDF Vectorizer and Naive Bayes classifier.
- Pretrained model provided for quick classification.

## Requirements
- pandas
- numpy
- scikit-learn
- nltk

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/phishing-email-classifier.git
   cd phishing-email-classifier
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. To classify a new email:
   ```python
   from phishing_email_classifier import classify_email
   print(classify_email("Your email content here"))
   ```

## Usage Example
```python
new_email = "Congratulations! You've won a prize. Click here to claim it."
print(classify_email(new_email))  # Output: Phishing
```
