# Sign Language Detection

This project is a real-time American Sign Language (ASL) alphabet detection system using Convolutional Neural Networks (CNNs).  
It allows recognition of hand gestures captured via webcam and maps them to the corresponding alphabet.

## Features
- Real-time sign language alphabet detection
- Pre-trained deep learning model (Keras/TensorFlow)
- Simple and clean interface
- Extendable for word/phrase detection

## Installation
Clone the repository:
```bash
git clone https://github.com/HetShingala/sign-language-detection.git
cd sign-language-detection
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the detection script:
```bash
python sign_language_detection.py
```

## Project Structure
- `sign_language_detection.py` – main detection script
- `requirements.txt` – dependencies
- `README.md` – project documentation
- Model files (`.h5`) and dictionary (`words_alpha.txt`) should be placed in the project directory.

## Future Scope
- Add word-level and sentence-level detection
- Improve model accuracy with larger datasets
- Develop a web or mobile app interface

## Author
Het Shingala
