# Stock Market News Analysis Project

This project analyzes the relationship between news headlines and stock market movements using the dataset from Kaggle: [Daily News for Stock Market Prediction](https://www.kaggle.com/datasets/aaron7sun/stocknews/data).

## Project Structure

```
├── src/
│   ├── data_loader.py    # Handles data loading and preprocessing
│   └── news_analyzer.py  # Contains news analysis functionality
├── data/                 # Directory for storing the dataset
├── main.py              # Main entry point of the application
└── README.md            # This file
```

## Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

2. Install required packages:
   ```
   pip install pandas numpy scikit-learn requests beautifulsoup4 matplotlib
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/aaron7sun/stocknews/data) and place it in the `data` directory.

## Usage

Run the main script:
```
python main.py
```

## Run the app
C:\dev\Teluq\SCI1402\.venv\Scripts\streamlit.exe run app.py

C:/dev/Teluq/SCI1402/.venv/Scripts/python.exe main.py

## Features

- Load and preprocess stock market news data
- Analyze news sentiment using TF-IDF vectorization
- Prepare data for machine learning models
- Basic analysis of news impact on stock market movements