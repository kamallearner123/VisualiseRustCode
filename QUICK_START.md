# Quick Start Guide - Updated Application

## What's New?

The application now supports **both Rust and Python programming** with a new home page!

## How to Use

### 1. Start the Application
```bash
./start.sh
# OR
python manage.py runserver
```

### 2. Open in Browser
Navigate to: http://127.0.0.1:8000/

### 3. Choose Your Language

#### Option A: Python Programming
- Click on **"Start with Python"** card
- Features:
  - 30+ Machine Learning & Data Science examples
  - Categories include:
    - Machine Learning (Linear Regression, SVM, Decision Trees, etc.)
    - Deep Learning (Neural Networks, CNN, RNN)
    - Data Science (Pandas, Data Cleaning, Visualization)
    - NumPy (Arrays, Matrices, Linear Algebra)
    - Python Basics (Variables, Functions, Classes, etc.)
  
- **How to Load Examples:**
  1. Click "Load Example" button
  2. Browse examples by category
  3. Click on any example to load it
  4. Click "Run Code" to execute

#### Option B: Rust Programming
- Click on **"Start with Rust"** card
- Features (existing):
  - Visual Memory Debugger with Miri
  - Ownership & Borrowing Visualization
  - 20+ Rust examples
  - Real-time Memory Tracing

## Python Examples Included

### Machine Learning Basics (6 examples)
- Linear Regression with scikit-learn
- Logistic Regression for classification
- Decision Trees
- Random Forest
- K-Means Clustering
- Support Vector Machines (SVM)

### Deep Learning (4 examples)
- Basic Neural Networks
- CNN for image classification
- RNN for time series
- Transfer Learning

### Data Science (5 examples)
- Pandas DataFrame operations
- Data cleaning techniques
- Data visualization
- Statistical analysis
- Feature engineering

### NumPy & Arrays (4 examples)
- Array operations
- Matrix operations
- Broadcasting
- Linear Algebra

### Python Basics (5 examples)
- Variables & data types
- Lists & dictionaries
- Functions & lambdas
- Classes & OOP
- File I/O

## Keyboard Shortcuts

### Python Editor
- `Ctrl/Cmd + Enter`: Run code
- `Ctrl/Cmd + K`: Clear editor and output

### Rust Editor
- `Ctrl/Cmd + Enter`: Run code with Miri
- `Ctrl/Cmd + K`: Clear editor and output

## Requirements

### For Python:
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- SciPy

Install with:
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### For Rust:
- Rust toolchain
- Miri (for memory debugging)

## Navigation
- Use the **back arrow** button in the top-left to return to the home page
- You can switch between Rust and Python at any time

## Tips
1. **Python Examples**: Start with "Python Basics" if you're new to Python
2. **ML Examples**: Try "Linear Regression" first for Machine Learning
3. **Execution Time**: Python code has a 30-second timeout
4. **Code Editing**: Full Monaco editor with syntax highlighting and auto-completion

## Troubleshooting

### Python code doesn't run
- Check that Python 3 is installed: `python3 --version`
- Install required packages: `pip install -r requirements.txt`

### Examples don't load
- Clear browser cache
- Check browser console for errors (F12)

### Server won't start
- Make sure port 8000 is available
- Check that virtual environment is activated

## What to Try First

1. **New to Machine Learning?**
   - Go to Python → Load Example → Machine Learning Basics → Linear Regression
   
2. **Want to explore Data Science?**
   - Go to Python → Load Example → Data Science → Pandas DataFrame Basics
   
3. **Learning Rust?**
   - Go to Rust → Load Example → Ownership - Move Semantics

Enjoy coding!
