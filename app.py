import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Diabetes Health Indicators Classifier",
    page_icon="🏥",
    layout="centered"
)

# Load the saved model and its classification threshold
bundle = joblib.load("models/optimized_logistic_regression.pkl")
model = bundle["model"]
threshold = bundle["threshold"]

# Keep the feature order exactly as used during model training
FEATURES = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "BMI",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "GenHlth",
    "MentHlth",
    "PhysHlth",
    "DiffWalk",
    "Sex",
    "Age",
    "Education",
    "Income"
]

st.title("Diabetes Health Indicators Classification System")

st.write(
    "An educational machine-learning demonstration using "
    "the CDC Diabetes Health Indicators dataset."
)

st.warning(
    "This tool is for educational purposes only. It is not a medical "
    "diagnosis or a substitute for professional healthcare. The model "
    "has not been validated for the Nigerian population."
)

st.subheader("Enter the survey information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        high_bp = st.selectbox("High blood pressure?", ["No", "Yes"])
        high_chol = st.selectbox("High cholesterol?", ["No", "Yes"])
        chol_check = st.selectbox(
            "Cholesterol checked in the last 5 years?",
            ["No", "Yes"]
        )
        bmi = st.number_input(
            "BMI",
            min_value=12,
            max_value=98,
            value=25
        )
        smoker = st.selectbox(
            "Have you smoked at least 100 cigarettes in your life?",
            ["No", "Yes"]
        )
        stroke = st.selectbox(
            "Have you ever had a stroke?",
            ["No", "Yes"]
        )
        heart_disease = st.selectbox(
            "Heart disease or heart attack?",
            ["No", "Yes"]
        )
        physical_activity = st.selectbox(
            "Any physical activity outside your job?",
            ["No", "Yes"]
        )
        fruits = st.selectbox(
            "Consume fruit at least once per day?",
            ["No", "Yes"]
        )
        veggies = st.selectbox(
            "Consume vegetables at least once per day?",
            ["No", "Yes"]
        )
        heavy_alcohol = st.selectbox(
            "Heavy alcohol consumption?",
            ["No", "Yes"]
        )

    with col2:
        healthcare = st.selectbox(
            "Have any kind of healthcare coverage?",
            ["No", "Yes"]
        )
        no_doctor_cost = st.selectbox(
            "Could not see a doctor due to cost in the past 12 months?",
            ["No", "Yes"]
        )
        general_health = st.selectbox(
            "How would you describe your general health?",
            ["Excellent", "Very good", "Good", "Fair", "Poor"]
        )
        mental_health = st.number_input(
            "Days mental health was not good in the past 30 days",
            min_value=0,
            max_value=30,
            value=0
        )
        physical_health = st.number_input(
            "Days physical health was not good in the past 30 days",
            min_value=0,
            max_value=30,
            value=0
        )
        difficulty_walking = st.selectbox(
            "Serious difficulty walking or climbing stairs?",
            ["No", "Yes"]
        )
        sex = st.selectbox(
            "Sex (as recorded in the dataset)",
            ["Female", "Male"]
        )
        age = st.selectbox(
            "Age group",
            [
                "18-24", "25-29", "30-34", "35-39", "40-44",
                "45-49", "50-54", "55-59", "60-64", "65-69",
                "70-74", "75-79", "80+"
            ]
        )
        education = st.selectbox(
            "Education level",
            [
                "Never attended / Kindergarten",
                "Grades 1-8",
                "Grades 9-11",
                "Grade 12 / GED",
                "College 1-3 years",
                "College 4+ years"
            ]
        )
        income = st.selectbox(
            "Annual household income (dataset categories)",
            [
                "Below $10,000",
                "$10,000-$14,999",
                "$15,000-$19,999",
                "$20,000-$24,999",
                "$25,000-$34,999",
                "$35,000-$49,999",
                "$50,000-$74,999",
                "$75,000 or more"
            ]
        )

    submitted = st.form_submit_button(
        "Evaluate responses",
        use_container_width=True
    )

if submitted:
    def yes_no(value):
        return 1 if value == "Yes" else 0

    age_options = [
        "18-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69",
        "70-74", "75-79", "80+"
    ]

    education_options = [
        "Never attended / Kindergarten",
        "Grades 1-8",
        "Grades 9-11",
        "Grade 12 / GED",
        "College 1-3 years",
        "College 4+ years"
    ]

    income_options = [
        "Below $10,000",
        "$10,000-$14,999",
        "$15,000-$19,999",
        "$20,000-$24,999",
        "$25,000-$34,999",
        "$35,000-$49,999",
        "$50,000-$74,999",
        "$75,000 or more"
    ]

    health_codes = {
        "Excellent": 1,
        "Very good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5
    }

    input_data = {
        "HighBP": yes_no(high_bp),
        "HighChol": yes_no(high_chol),
        "CholCheck": yes_no(chol_check),
        "BMI": bmi,
        "Smoker": yes_no(smoker),
        "Stroke": yes_no(stroke),
        "HeartDiseaseorAttack": yes_no(heart_disease),
        "PhysActivity": yes_no(physical_activity),
        "Fruits": yes_no(fruits),
        "Veggies": yes_no(veggies),
        "HvyAlcoholConsump": yes_no(heavy_alcohol),
        "AnyHealthcare": yes_no(healthcare),
        "NoDocbcCost": yes_no(no_doctor_cost),
        "GenHlth": health_codes[general_health],
        "MentHlth": mental_health,
        "PhysHlth": physical_health,
        "DiffWalk": yes_no(difficulty_walking),
        "Sex": 0 if sex == "Female" else 1,
        "Age": age_options.index(age) + 1,
        "Education": education_options.index(education) + 1,
        "Income": income_options.index(income) + 1
    }

    input_df = pd.DataFrame([input_data], columns=FEATURES)

    score = model.predict_proba(input_df)[0][1]
    prediction = int(score >= threshold)

    st.subheader("Model result")
    st.metric("Model score", f"{score:.1%}")

    if prediction == 1:
        st.warning(
            "The model classified these responses in the higher-likelihood "
            "category for reported prediabetes or diabetes."
        )
    else:
        st.success(
            "The model classified these responses in the lower-likelihood "
            "category for reported prediabetes or diabetes."
        )

    st.caption(
        "This score is a model estimate, not a clinically validated personal "
        "probability. The classification cutoff was selected during model "
        "validation. A lower score does not rule out diabetes."
    )

with st.expander("About this project and its limitations"):
    st.write(
        "- The model uses the CDC Diabetes Health Indicators survey dataset.\n"
        "- The target combines reported prediabetes and diabetes.\n"
        "- The dataset reflects a US survey population.\n"
        "- The model has not been clinically validated.\n"
        "- Survey patterns do not establish cause and effect.\n"
        "- A lower model score does not rule out diabetes."
    )
