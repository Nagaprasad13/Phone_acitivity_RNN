import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ==============================
# 🎯 Load Artifacts
# ==============================
# Make sure these files exist in the same folder:
#   - model_lstm.h5
#   - scaler.pkl
#   - label_encoder.pkl

model = load_model("model_lstm.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ==============================
# 🧭 Streamlit UI
# ==============================
st.set_page_config(page_title="LSTM Activity Predictor", page_icon="🤖", layout="centered")

st.title("🤖 Human Activity Prediction using LSTM")
st.markdown("Enter feature values to predict the **Activity** class.")

# Dynamically build input fields
# NOTE: change `num_features` to your actual number of features
num_features = scaler.mean_.shape[0]

user_input = []
cols = st.columns(3)
for i in range(num_features):
    col = cols[i % 3]
    val = col.number_input(f"Feature {i+1}", value=0.0, format="%.3f")
    user_input.append(val)

# ==============================
# ⚙️ Predict Button
# ==============================
if st.button("Predict Activity 🚀"):
    # Convert input to numpy array
    input_array = np.array(user_input).reshape(1, -1)

    # Scale using fitted scaler
    input_scaled = scaler.transform(input_array)

    # Reshape for LSTM (samples, timesteps, features)
    input_reshaped = np.expand_dims(input_scaled, axis=1)

    # Predict
    prediction = model.predict(input_reshaped)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = le.inverse_transform([predicted_class])[0]

    st.success(f"🎯 Predicted Activity: **{predicted_label}**")
    st.write("Confidence:", f"{np.max(prediction)*100:.2f}%")

# ==============================
# 🌈 Footer
# ==============================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow LSTM.")
