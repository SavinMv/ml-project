import joblib
from preprocess import preprocess_text

model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_spam(message):
    """Predict if the message is spam or not."""
    processed_msg = preprocess_text(message)
    vectorized_msg = vectorizer.transform([processed_msg]).toarray()
    prediction = model.predict(vectorized_msg)
    return "Spam" if prediction[0] == 1 else "Not Spam"

test_msg1 = "Congratulations! You won a free iPhone. Click here to claim."
test_msg2 = "Hey, are we meeting for dinner tonight?"

print(f"Message: '{test_msg1}' -> {predict_spam(test_msg1)}")
print(f"Message: '{test_msg2}' -> {predict_spam(test_msg2)}")
