```python
import streamlit as st
import pandas as pd
import pickle


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("heart_disease_xgb_pipeline.pkl", "rb") as file:
        model = pickle.load(file)

    return model


# ============================================================
# TRY TO LOAD MODEL
# ============================================================

try:

    model = load_model()

    st.success("✅ Model loaded successfully!")

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.write("### Actual Error")

    st.code(str(e))

    st.write("### Error Type")

    st.code(type(e).__name__)

    st.info(
        """
        The model file exists, but Streamlit may be unable to load it
        because of a Python, scikit-learn, or XGBoost compatibility issue.

        Send the error shown above if this message appears.
        """
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    """
    Enter the patient's information below.

    The trained XGBoost machine-learning model will estimate whether
    the patient is likely to have heart disease.
    """
)


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.warning(
    """
    ⚠️ Medical Disclaimer

    This application is for educational and research purposes only.

    It is NOT a medical diagnosis system and should not replace
    evaluation by a qualified healthcare professional.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("❤️ Heart Disease AI")

st.sidebar.write(
    """
    ### Machine Learning Model

    **Algorithm:** XGBoost

    **Task:** Binary Classification

    **Class 0:** No Heart Disease

    **Class 1:** Heart Disease

    **Preprocessing:** One-Hot Encoding

    **Deployment:** Streamlit Community Cloud
    """
)

st.sidebar.divider()

st.sidebar.info(
    """
    ### Model Performance

    Test Accuracy:

    **77.05%**

    Test ROC-AUC:

    **87.45%**
    """
)


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.header("🧑 Patient Information")

st.write(
    "Please enter the patient's information below."
)


# ============================================================
# ROW 1
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50,
        step=1
    )


with col2:

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x:
        "Female (0)" if x == 0 else "Male (1)"
    )


with col3:

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Typical Angina",
            1: "1 - Atypical Angina",
            2: "2 - Non-anginal Pain",
            3: "3 - Asymptomatic"
        }[x]
    )


# ============================================================
# ROW 2
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )


with col2:

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200,
        step=1
    )


with col3:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
    )


# ============================================================
# ROW 3
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Normal",
            1: "1 - ST-T Wave Abnormality",
            2: "2 - Left Ventricular Hypertrophy"
        }[x]
    )


with col2:

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )


with col3:

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda x:
        "No (0)" if x == 0 else "Yes (1)"
    )


# ============================================================
# ROW 4
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


with col2:

    slope = st.selectbox(
        "ST Segment Slope",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "0 - Upsloping",
            1: "1 - Flat",
            2: "2 - Downsloping"
        }[x]
    )


with col3:

    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        options=[0, 1, 2, 3, 4]
    )


# ============================================================
# ROW 5
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    thal = st.selectbox(
        "Thalassemia (thal)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0",
            1: "1",
            2: "2",
            3: "3"
        }[x]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Heart Disease",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "age": [age],

        "sex": [sex],

        "cp": [cp],

        "trestbps": [trestbps],

        "chol": [chol],

        "fbs": [fbs],

        "restecg": [restecg],

        "thalach": [thalach],

        "exang": [exang],

        "oldpeak": [oldpeak],

        "slope": [slope],

        "ca": [ca],

        "thal": [thal]

    })


    # --------------------------------------------------------
    # SHOW INPUT DATA
    # --------------------------------------------------------

    st.subheader("📋 Patient Information")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        no_disease_probability = probabilities[0]

        disease_probability = probabilities[1]


        # ====================================================
        # RESULT SECTION
        # ====================================================

        st.divider()

        st.header("📊 Prediction Result")


        # ====================================================
        # HEART DISEASE
        # ====================================================

        if prediction == 1:

            st.error(
                "⚠️ Model Prediction: Heart Disease"
            )

            st.metric(
                label="Heart Disease Probability",
                value=f"{disease_probability * 100:.2f}%"
            )

            st.write(
                f"""
                ### ⚠️ Model Assessment

                Based on the information entered, the model classified
                this patient as:

                **Heart Disease — Class 1**

                Estimated model probability:

                **{disease_probability * 100:.2f}%**

                Probability of No Heart Disease:

                **{no_disease_probability * 100:.2f}%**
                """
            )

            st.warning(
                """
                This prediction does NOT confirm that the patient has
                heart disease.

                It is only the output of a machine-learning model.

                A qualified healthcare professional should evaluate
                the patient and perform appropriate medical tests.
                """
            )


        # ====================================================
        # NO HEART DISEASE
        # ====================================================

        else:

            st.success(
                "✅ Model Prediction: No Heart Disease"
            )

            st.metric(
                label="No Heart Disease Probability",
                value=f"{no_disease_probability * 100:.2f}%"
            )

            st.write(
                f"""
                ### ✅ Model Assessment

                Based on the information entered, the model classified
                this patient as:

                **No Heart Disease — Class 0**

                Estimated model probability:

                **{no_disease_probability * 100:.2f}%**

                Probability of Heart Disease:

                **{disease_probability * 100:.2f}%**
                """
            )

            st.info(
                """
                This prediction does NOT guarantee that the patient
                is free from heart disease.

                If the patient has symptoms or health concerns,
                they should consult a qualified healthcare professional.
                """
            )


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        st.subheader("📈 Prediction Probability")

        probability_df = pd.DataFrame(
            {
                "Condition": [
                    "No Heart Disease",
                    "Heart Disease"
                ],

                "Probability": [
                    no_disease_probability,
                    disease_probability
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index("Condition")
        )


        # ====================================================
        # PROBABILITY TABLE
        # ====================================================

        st.subheader("📊 Probability Details")

        probability_display = pd.DataFrame(
            {
                "Condition": [
                    "No Heart Disease",
                    "Heart Disease"
                ],

                "Probability": [
                    f"{no_disease_probability * 100:.2f}%",
                    f"{disease_probability * 100:.2f}%"
                ]
            }
        )

        st.table(probability_display)


        # ====================================================
        # MODEL EXPLANATION
        # ====================================================

        st.divider()

        st.subheader("🤖 How the AI Model Works")

        st.write(
            """
            The application uses the same preprocessing and XGBoost
            model that were used during training.

            The process is:

            1. The user enters patient information.

            2. Streamlit creates a Pandas DataFrame.

            3. The saved Pipeline receives the data.

            4. One-Hot Encoding transforms categorical features.

            5. XGBoost analyzes the transformed data.

            6. The model predicts Class 0 or Class 1.

            7. The application displays the prediction probability.
            """
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.subheader("🧠 Model Information")

        model_info_col1, model_info_col2, model_info_col3 = st.columns(3)


        with model_info_col1:

            st.metric(
                "Algorithm",
                "XGBoost"
            )


        with model_info_col2:

            st.metric(
                "Test Accuracy",
                "77.05%"
            )


        with model_info_col3:

            st.metric(
                "Test ROC-AUC",
                "87.45%"
            )


    # ========================================================
    # PREDICTION ERROR
    # ========================================================

    except Exception as e:

        st.error(
            "❌ An error occurred while making the prediction."
        )

        st.write("### Actual Prediction Error")

        st.code(str(e))

        st.write("### Error Type")

        st.code(type(e).__name__)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    ❤️ Heart Disease Prediction | XGBoost Machine Learning Project

    Educational and research use only — not a medical diagnosis.
    """
)
```
