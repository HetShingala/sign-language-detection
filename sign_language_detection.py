import cv2
import time
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/Users/Het/Desktop/AI/asl_model.h5')

# Labels
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Load simple word list for prediction
with open('words_alpha.txt') as f:  # Download "words_alpha.txt" from internet (big English dictionary)
    words = set(f.read().split())

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Start capturing video
cap = cv2.VideoCapture(0)

# Store captured letters
captured_message = ""

def predict_word(prefix):
    """Predict a full word based on current prefix"""
    suggestions = [word for word in words if word.startswith(prefix.lower())]
    return suggestions[0] if suggestions else ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    roi = cv2.resize(frame, (64, 64))
    roi = np.expand_dims(roi / 255.0, axis=0)

    # Predict
    pred = model.predict(roi)
    class_id = np.argmax(pred)
    predicted_label = labels[class_id]

    # Word prediction (after 2 letters typed)
    current_word = captured_message.split(' ')[-1]  # last word
    predicted_word = predict_word(current_word) if len(current_word) >= 2 else ""

    # Show current prediction
    cv2.putText(frame, f"Current: {predicted_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show captured message so far
    cv2.putText(frame, f"Message: {captured_message}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show predicted word
    if predicted_word:
        cv2.putText(frame, f"Suggestion: {predicted_word}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Sign Detection', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # If 'c' is pressed, capture current letter
        if predicted_label == 'space':
            captured_message += ' '
        elif predicted_label == 'del':
            captured_message = captured_message[:-1]
        elif predicted_label == 'nothing':
            pass
        else:
            captured_message += predicted_label
        print(f"Captured Message So Far: {captured_message}")

    if key == ord('p'):
        # If 'p' is pressed, complete current word with prediction
        if predicted_word:
            parts = captured_message.split(' ')
            parts[-1] = predicted_word  # Replace last word with prediction
            captured_message = ' '.join(parts)
            print(f"Auto-completed to: {captured_message}")

    if key == ord('v'):
        # If 'v' is pressed, speak the current message
        engine.say(captured_message)
        engine.runAndWait()

    if key == ord('q'):
        # If 'q' is pressed, quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final message
print("Final Message:", captured_message)
