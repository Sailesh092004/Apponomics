import streamlit as st
import pandas as pd

st.title("Apponomics Predictions")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.write(df)

    def predict_tier(spend: float) -> str:
        if spend >= 500:
            return "premium"
        if spend >= 100:
            return "standard"
        return "free"

    def predict_spend(spend: float) -> float:
        return spend * 1.1

    def predict_churn(spend: float, sessions: float) -> float:
        if sessions < 3 or spend < 50:
            return 0.8
        return 0.2

    def predict_cluster(spend: float, sessions: float) -> str:
        if spend >= 500 or sessions >= 10:
            return "A"
        if spend >= 100 or sessions >= 5:
            return "B"
        return "C"

    if 'spend' not in df.columns:
        st.error("CSV must contain a 'spend' column.")
    else:
        sessions_col = df.get('sessions', pd.Series([0] * len(df)))
        df['predicted_tier'] = df['spend'].apply(predict_tier)
        df['predicted_spend'] = df['spend'].apply(predict_spend)
        df['predicted_churn'] = [predict_churn(s, sess) for s, sess in zip(df['spend'], sessions_col)]
        df['predicted_cluster'] = [predict_cluster(s, sess) for s, sess in zip(df['spend'], sessions_col)]

        st.subheader("Predictions")
        st.write(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions", csv, "predictions.csv", "text/csv")
else:
    st.info("Please upload a CSV file to begin.")
