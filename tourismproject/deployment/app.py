import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Travel Conversion AI",
    page_icon="✈️",
    layout="centered"
)

# -----------------------------
# Download and Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="muthukumar22/tourism-package-mod",
        filename="best_tourism_model_rf.joblib" # Updated to match the RF model name
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# 1. New Title Section
# -----------------------------
st.title("✈️ Intelligent Travel Sales Predictor")

# -----------------------------
# 2. New Comments/Description Section
# -----------------------------
st.markdown("""
---
### 🎯 Optimize Your Sales Strategy
Welcome to the **Customer Conversion Dashboard**. This tool utilizes a **Random Forest** machine learning model to analyze customer demographics and interaction history.

**How to use:**
1. Input the customer's profile details below.
2. Click **Analyze Customer Potential**.
3. Receive a probability score indicating the likelihood of a sale.
---
""")

# -----------------------------
# User Inputs (Organized with Columns)
# -----------------------------
st.subheader("📝 Customer Profile")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 18, 80, 35)
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfPersonVisiting = st.number_input("Persons Visiting", 1, 10, 2)
    NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 5, 0)
    MonthlyIncome = st.number_input("Monthly Income", 10000, 200000, 50000)

with col2:
    TypeofContact = st.selectbox("Contact Method", ["Company Invited", "Self Inquiry"])
    Designation = st.selectbox("Job Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
    NumberOfTrips = st.number_input("Trips per Year", 0, 20, 2)
    Passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    OwnCar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.subheader("📞 Interaction Details")
col3, col4 = st.columns(2)
with col3:
    PitchSatisfactionScore = st.slider("Pitch Satisfaction (1-5)", 1, 5, 3)
with col4:
    NumberOfFollowups = st.number_input("Follow-ups Made", 0, 10, 3)
    DurationOfPitch = st.number_input("Pitch Duration (mins)", 1, 120, 20)

# -----------------------------
# Assemble Input Data
# -----------------------------
# Note: Keys must match the training column names exactly
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}])

# -----------------------------
# 3. New Prediction Section
# -----------------------------
st.markdown("---")
if st.button("Analyze Customer Potential", type="primary", use_container_width=True):
    
    # Get the probability of class 1 (Purchase)
    # Most sklearn classifiers support predict_proba
    try:
        prediction_prob = model.predict_proba(input_data)[0][1]
        prediction_class = model.predict(input_data)[0]
    except AttributeError:
        # Fallback if model doesn't support proba (unlikely for RF)
        prediction_class = model.predict(input_data)[0]
        prediction_prob = 1.0 if prediction_class == 1 else 0.0

    st.subheader("📊 Analysis Results")
    
    r_col1, r_col2 = st.columns([1, 2])

    with r_col1:
        # Display simple metric
        st.metric(label="Purchase Probability", value=f"{prediction_prob:.1%}")

    with r_col2:
        # Display logic with visuals
        if prediction_prob > 0.65:
            st.success("✅ **High Conversion Chance**")
            st.write("This customer shows strong interest signals. **Recommended Action:** Close the deal immediately.")
            st.progress(prediction_prob)
        elif prediction_prob > 0.35:
            st.warning("⚠️ **Medium Conversion Chance**")
            st.write("The customer is on the fence. **Recommended Action:** Offer a discount or follow up.")
            st.progress(prediction_prob)
        else:
            st.error("🔻 **Low Conversion Chance**")
            st.write("This customer is unlikely to purchase. **Recommended Action:** Do not prioritize.")
            st.progress(prediction_prob)
