# Setup Guide for Cat vs Dog Image Classifier

## Issue
Your current Python version (3.14) is not compatible with TensorFlow. TensorFlow requires Python 3.11 or 3.12.

## Solution

### Step 1: Install Python 3.12
1. Download Python 3.12 from [python.org](https://www.python.org/downloads/)
2. During installation, **check "Add Python to PATH"**
3. Install as "for all users" if possible

### Step 2: Create Virtual Environment with Python 3.12
```powershell
cd "c:\Users\ASUS\OneDrive\Documents\Image_Recognition_Project"
py -3.12 -m venv .venv
```

### Step 3: Activate Virtual Environment
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try activation again
```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Add Model File
Place the `image_classifier.keras` model file in the project directory

### Step 6: Run the App
```powershell
streamlit run app.py
```

## VS Code Configuration
1. Open Command Palette (Ctrl+Shift+P)
2. Search for "Python: Select Interpreter"
3. Choose the path ending in `.\.venv\Scripts\python.exe`

## Troubleshooting
- If `py -3.12` doesn't work, use `python3.12 -m venv .venv` instead
- Ensure the model file `image_classifier.keras` exists in the project directory
