# Fraud Data Explorer - Streamlit App

This project is an interactive Streamlit application for exploring and analyzing fraud-related datasets, including credit card fraud and bank transaction data. The app provides visualizations, summary statistics, and fraud-specific insights in a user-friendly web interface.

## Features
- Select between Credit Card Fraud Data and Bank Transaction Data
- View sample data and summary statistics
- Visualize transaction amount distributions and categorical breakdowns
- For credit card data: see fraud counts, fraud percentage, fraud vs. non-fraud pie chart, and sample fraudulent transactions
- For bank data: see transaction type and channel breakdowns, and top locations
- All visualizations are interactive and update based on dataset selection

## Setup

1. **Clone or download this repository.**
2. **Place your datasets** (`fraudTrain.csv` and `bank_transactions_data_2.csv`) in the project folder.
3. **(Optional) Create and activate a virtual environment:**
   ```
   python -m venv .venv
   .\.venv\Scripts\activate  # On Windows
   source .venv/bin/activate   # On Mac/Linux
   ```
4. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Running the App

```
streamlit run main.py
```

or, if the `streamlit` command is not recognized:
```
python -m streamlit run main.py
```

Then open the provided local URL in your browser (usually http://localhost:8501).

## Datasets
- **Credit Card Fraud Data:** `fraudTrain.csv` (must include an `is_fraud` column)
- **Bank Transaction Data:** `bank_transactions_data_2.csv`

## Requirements
- Python 3.7+
- See `requirements.txt` for Python package dependencies.

## Notes
- The app is designed for interactive data exploration and visualization. No command-line interface is provided.

- You can extend the app with more advanced analytics or machine learning as needed.

---


## Developed By

G.Yuva Kishore Reddy  
Passionate about AI, automation, and building impactful tech solutions.

## Contact

Email: g.yuvakishorereddy@gmail.com   
WhatsApp Channel: https://whatsapp.com/channel/0029Vb3la9V7NoZtA1GUI00d

Star⭐ this repo if you found it helpful!

