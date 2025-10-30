import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# =======================
# APP TITLE & INFO
# =======================
st.set_page_config(page_title="📱 Human Activity Recognition", layout="centered")
st.title("📱 Human Activity Recognition App (RNN Model)")
st.markdown("""
This app predicts **human activity** using smartphone sensor data.  
You can either input values manually or upload a dataset for batch predictions.
""")

# =======================
# LOAD MODEL
# =======================
MODEL_PATH = "model_lstm.h5"
model = load_model(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# =======================
# SIDEBAR MODE SELECTOR
# =======================
st.sidebar.header("⚙️ App Settings")
option = st.sidebar.radio(
    "Choose Prediction Mode:",
    ["🔹 Single Input", "🔸 Dataset Upload"]
)

# =======================
# SINGLE INPUT MODE
# =======================
if option == "🔹 Single Input":
    st.subheader("🧾 Enter Sensor Feature Values")
    st.markdown("Provide input values for each feature below:")

    # Replace with your model's actual input feature names
    feature_names = ['tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z']

    inputs = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, step=0.01)
        inputs.append(val)

    if st.button("🚀 Predict Activity"):
        X_input = np.array([inputs])  # shape (1, n_features)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_input)

        prediction = model.predict(X_scaled)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.success(f"🏃 Predicted Activity: **{predicted_class}**")
        st.balloons()

# =======================
# DATASET UPLOAD MODE
# =======================
else:
    st.subheader("📂 Upload Dataset for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Dataset Uploaded Successfully!")
        st.dataframe(df.head())

        if st.button("⚡ Predict on Dataset"):
            X = df.drop(columns=['Activity', 'subject'], errors='ignore')

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            preds = model.predict(X_scaled)
            predicted_classes = np.argmax(preds, axis=1)

            df['Predicted_Activity'] = predicted_classes
            st.success("🎯 Predictions Completed!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

# =======================
# FOOTER
# =======================
st.markdown("---")
st.caption("👨‍💻 Developed by Nagaprasad | Powered by Streamlit + TensorFlow")
