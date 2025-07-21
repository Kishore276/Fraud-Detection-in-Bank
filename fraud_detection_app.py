import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bank Fraud Detection", layout="wide")
st.title("Fraud Detection in Banking Data (Machine Learning Demo)")

@st.cache_data
def load_data():
    df = pd.read_csv("bank_transactions_data_2.csv")
    return df

df = load_data()
st.subheader("Raw Data Sample")
st.dataframe(df.head())

# --- Feature Engineering ---
st.subheader("Feature Engineering")

df_fe = df.copy()

# Convert dates
df_fe['TransactionDate'] = pd.to_datetime(df_fe['TransactionDate'])
df_fe['PreviousTransactionDate'] = pd.to_datetime(df_fe['PreviousTransactionDate'])

# Time since previous transaction (in hours)
df_fe['TimeSincePrevTx'] = (df_fe['TransactionDate'] - df_fe['PreviousTransactionDate']).dt.total_seconds() / 3600

# Transaction amount z-score (per account)
df_fe['Amount_zscore'] = df_fe.groupby('AccountID')['TransactionAmount'].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0)
)

# Transaction count per account
df_fe['TxCount_Account'] = df_fe.groupby('AccountID')['TransactionID'].transform('count')

# Average transaction amount per account
df_fe['AvgAmount_Account'] = df_fe.groupby('AccountID')['TransactionAmount'].transform('mean')

# Transaction count per device
df_fe['TxCount_Device'] = df_fe.groupby('DeviceID')['TransactionID'].transform('count')

# Drop columns not useful for ML
drop_cols = ['TransactionID', 'AccountID', 'DeviceID', 'IP Address', 'MerchantID', 'TransactionDate', 'PreviousTransactionDate']
df_fe = df_fe.drop(columns=drop_cols)

# Encode categoricals
cat_cols = df_fe.select_dtypes(include=['object']).columns
for col in cat_cols:
    df_fe[col] = LabelEncoder().fit_transform(df_fe[col].astype(str))

# Fill missing values
df_fe = df_fe.fillna(0)

# --- Synthetic Fraud Label ---
st.sidebar.header("Synthetic Fraud Label Settings")
fraud_ratio = st.sidebar.slider("Fraud Ratio (for demo)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
np.random.seed(42)
df_fe['is_fraud'] = np.random.choice([0, 1], size=len(df_fe), p=[1-fraud_ratio, fraud_ratio])

# --- Train/Test Split ---
X = df_fe.drop('is_fraud', axis=1)
y = df_fe['is_fraud']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model Training ---
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# --- Evaluation ---
st.subheader("Model Evaluation")

col1, col2 = st.columns(2)
with col1:
    st.write("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col2:
    st.write("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
st.write("**ROC Curve**")
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax2.plot([0, 1], [0, 1], "k--")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("Receiver Operating Characteristic")
ax2.legend(loc="lower right")
st.pyplot(fig2)

# Feature Importances
st.write("**Feature Importances**")
importances = pd.Series(clf.feature_importances_, index=X.columns)
st.bar_chart(importances.sort_values(ascending=False))

st.success("Demo complete! Replace the synthetic label with real fraud data for production use.")

st.markdown("""
**Tips for improvement:**
- Replace the synthetic label with your real fraud labels if available.
- Try more advanced models (XGBoost, LightGBM, etc.).
- Add more domain-specific features (e.g., velocity, geolocation anomalies).
- Use cross-validation for more robust evaluation.
""")