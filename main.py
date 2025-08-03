import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Data Explorer", layout="wide")
st.title("Fraud Data Explorer")

DATASETS = {
    "Credit Card Fraud Data": "fraudTrain.csv",
    "Bank Transaction Data": "bank_transactions_data_2.csv"
}

dataset_name = st.sidebar.radio("Select dataset to analyze:", list(DATASETS.keys()))
file_path = DATASETS[dataset_name]

st.header(f"Selected Dataset: {dataset_name}")

@st.cache_data
def load_data(path, nrows=None):
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return None

nrows = 100000 if "Credit Card" in dataset_name else None
df = load_data(file_path, nrows=nrows)

if df is not None:
    st.markdown(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
    st.subheader("Sample Data")
    if dataset_name == "Credit Card Fraud Data" and "is_fraud" in df.columns:
        fraud_sample = df[df['is_fraud'] == 1].head(5)
        if not fraud_sample.empty:
            st.dataframe(fraud_sample)
        else:
            st.dataframe(df.head())
    else:
        st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    # Visualizations
    st.subheader("Visualizations")
    # Transaction Amount Distribution
    amt_col = None
    if dataset_name == "Credit Card Fraud Data":
        amt_col = "amt"
    elif dataset_name == "Bank Transaction Data":
        amt_col = "TransactionAmount"
    if amt_col and amt_col in df.columns:
        st.markdown(f"**Distribution of {amt_col}:**")
        fig, ax = plt.subplots()
        sns.histplot(df[amt_col], bins=50, kde=True, ax=ax)
        ax.set_xlabel(amt_col)
        st.pyplot(fig)

    # Categorical column bar plot
    cat_col = None
    if dataset_name == "Credit Card Fraud Data":
        cat_col = "category"
    elif dataset_name == "Bank Transaction Data":
        cat_col = "TransactionType"
    if cat_col and cat_col in df.columns:
        st.markdown(f"**Counts of {cat_col}:**")
        fig2, ax2 = plt.subplots()
        df[cat_col].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_xlabel(cat_col)
        ax2.set_ylabel('Count')
        st.pyplot(fig2)

    # Credit Card Fraud-specific analysis
     # Credit Card Fraud-specific analysis
    if dataset_name == "Credit Card Fraud Data" and "is_fraud" in df.columns:
        st.subheader("Fraud Analysis")
        fraud_count = df["is_fraud"].sum()
        fraud_pct = 100 * df["is_fraud"].mean()
        st.info(f"Fraudulent Transactions: {fraud_count} ({fraud_pct:.4f}%)")

        # Pie chart of fraud vs non-fraud
        st.markdown("**Fraud vs. Non-Fraud Transactions:**")
        fig3, ax3 = plt.subplots()
        df['is_fraud'].value_counts().plot.pie(labels=['Non-Fraud', 'Fraud'], autopct='%1.2f%%', startangle=90, ax=ax3)
        ax3.set_ylabel('')
        st.pyplot(fig3)

        # Show fraud transaction details
        st.markdown("**Sample Fraudulent Transactions:**")
        st.dataframe(df[df['is_fraud'] == 1].head(10))

        # Fraud by category
        if 'category' in df.columns:
            st.markdown("**Fraud Count by Category:**")
            fraud_by_cat = df[df['is_fraud'] == 1]['category'].value_counts()
            st.bar_chart(fraud_by_cat)

        # Fraud by amount
        st.markdown("**Fraudulent Transaction Amounts Distribution:**")
        fig4, ax4 = plt.subplots()
        sns.histplot(df[df['is_fraud'] == 1][amt_col], bins=30, kde=True, color='red', ax=ax4)
        ax4.set_xlabel(amt_col)
        st.pyplot(fig4)

    # Bank Data: show TransactionType distribution if available
    if dataset_name == "Bank Transaction Data" and "TransactionType" in df.columns:
        st.markdown("**Transaction Type Distribution:**")
        st.bar_chart(df['TransactionType'].value_counts())

    # Add more useful visualizations as needed
    # For both: Transaction amount by channel/merchant if available
    if dataset_name == "Bank Transaction Data" and "Channel" in df.columns:
        st.markdown("**Average Transaction Amount by Channel:**")
        avg_amt_by_channel = df.groupby('Channel')['TransactionAmount'].mean()
        st.bar_chart(avg_amt_by_channel)

    if dataset_name == "Bank Transaction Data" and "Location" in df.columns:
        st.markdown("**Top 10 Locations by Number of Transactions:**")
        top_locs = df['Location'].value_counts().head(10)
        st.bar_chart(top_locs)
else:
    st.warning("No data loaded. Please check your file path or file format.")

st.markdown("---")

st.caption("Developed for interactive fraud data analysis.") 
