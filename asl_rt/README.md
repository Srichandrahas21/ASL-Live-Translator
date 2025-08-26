# ASL Letters — Real-Time Sign Translator (Webcam + MediaPipe)

Real-time ASL fingerspelling (A–Z) from webcam using MediaPipe Hands + scikit-learn.

## Demo
- Train: `.\.venv311\Scripts\python.exe src\train_letters.py`
- Inference: `.\.venv311\Scripts\python.exe src\infer_letters.py`

## Setup (Windows, Python 3.11)
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
