import joblib
import re
import string

# Load saved files
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("model.pkl")

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

while True:
    msg = input("\nEnter Message (type 'exit' to quit): ")

    if msg.lower() == "exit":
        break

    msg = clean_text(msg)

    msg_vector = vectorizer.transform([msg])

    prediction = model.predict(msg_vector)

    print("Prediction:", prediction[0].upper())