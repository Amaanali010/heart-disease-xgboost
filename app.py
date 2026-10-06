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
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open("heart_disease_xgb_pipeline.pkl", "rb") as file:
        model = pickle.load(file)

    return model


try:
    model = load_model()

except Exception as e:

    st.error("❌ Model could not be loaded.")

    st.write("Make sure this file exists in your GitHub repository:")

    st.code("heart_disease_xgb_pipeline.pkl")

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    """
    Enter the patient's information below. The trained XGBoost
    machine-learning model will estimate whether the patient is
    likely to have heart disease.
    """
)

st.warning(
    "⚠️ This application is for educational and research purposes only. "
    "It is NOT a medical diagnosis and should not replace a qualified doctor."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("About This App")

st.sidebar.write(
    """
    This application uses an XGBoost classification model.

    The model was trained using heart disease patient data.

    Model:
    XGBoost

    Preprocessing:
    One-Hot Encoding

    Deployment:
    Streamlit Community Cloud
    """
)

st.sidebar.info(
    """
    Prediction classes:

    0 = No Heart Disease

    1 = Heart Disease
    """
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🧑 Patient Information")


# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

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
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {

            0: "0 - Typical Angina",
            1: "1 - Atypical Angina",
            2: "2 - Non-anginal Pain",
            3: "3 - Asymptomatic"

        }[x]
    )


# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    trestbps = st.number_input(
        "Resting Blood Pressure (trestbps)",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )


with col2:

    chol = st.number_input(
        "Cholesterol (chol)",
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


# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    restecg = st.selectbox(
        "Resting ECG (restecg)",
        options=[0, 1, 2],
        format_func=lambda x: {

            0: "0 - Normal",
            1: "1 - ST-T Wave Abnormality",
            2: "2 - Left Ventricular Hypertrophy"

        }[x]
    )


with col2:

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)",
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


# ------------------------------------------------------------
# ROW 4
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )


with col2:

    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
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


# ------------------------------------------------------------
# ROW 5
# ------------------------------------------------------------

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
    # CREATE DATAFRAME
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
    # MAKE PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        no_disease_probability = probability[0]

        disease_probability = probability[1]


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.header("📊 Prediction Result")


        # ----------------------------------------------------
        # HEART DISEASE
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Heart Disease Detected"
            )

            st.metric(
                "Estimated Heart Disease Probability",
                f"{disease_probability * 100:.2f}%"
            )

            st.write(
                f"""
                ### ⚠️ Model Result

                The model predicts **Heart Disease (Class 1)**.

                Estimated probability:

                **{disease_probability * 100:.2f}%**

                The model's estimated probability of no heart disease is:

                **{no_disease_probability * 100:.2f}%**
                """
            )

            st.warning(
                """
                This does NOT mean the person definitely has heart disease.

                The result should be discussed with a qualified healthcare
                professional and confirmed using appropriate clinical
                evaluation and medical tests.
                """
            )


        # ----------------------------------------------------
        # NO HEART DISEASE
        # ----------------------------------------------------

        else:

            st.success(
                "✅ Prediction: No Heart Disease Detected"
            )

            st.metric(
                "Estimated No Heart Disease Probability",
                f"{no_disease_probability * 100:.2f}%"
            )

            st.write(
                f"""
                ### ✅ Model Result

                The model predicts **No Heart Disease (Class 0)**.

                Estimated probability:

                **{no_disease_probability * 100:.2f}%**

                The model's estimated probability of heart disease is:

                **{disease_probability * 100:.2f}%**
                """
            )

            st.info(
                """
                A prediction of no heart disease does not guarantee that
                the person is healthy. If symptoms or concerns exist,
                consult a qualified healthcare professional.
                """
            )


        # ====================================================
        # PROBABILITY CHART
        # ====================================================

        st.subheader("📈 Prediction Probability")

        probability_df = pd.DataFrame({

            "Condition": [
                "No Heart Disease",
                "Heart Disease"
            ],

            "Probability": [
                no_disease_probability,
                disease_probability
            ]

        })

        st.bar_chart(
            probability_df.set_index("Condition")
        )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader("📝 Patient Input Summary")

        display_data = input_data.copy()

        display_data.columns = [

            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Angina",
            "ST Depression",
            "ST Slope",
            "Major Vessels",
            "Thalassemia"

        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # MODEL EXPLANATION
        # ====================================================

        st.subheader("🤖 How This Prediction Works")

        st.write(
            """
            The application sends the information you entered to the
            trained XGBoost machine-learning model.

            The model first applies the same preprocessing used during
            training. Categorical variables are transformed using
            One-Hot Encoding.

            XGBoost then analyzes the input features and produces:

            • Class 0 → No Heart Disease

            • Class 1 → Heart Disease

            The model also produces probabilities for both classes.
            """
        )


    except Exception as e:

        st.error("❌ An error occurred while making the prediction.")

        st.code(str(e))
