import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Set page layout to wide mode
st.set_page_config(layout="wide")

# Load the pre-trained model
with open('random_forest_model_frailty.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Page title
st.title("Stroke Frailty Prediction Model")

# Create two columns for layout
left_col, right_col = st.columns([1, 1.5])  # Adjusting column width ratios

# Feature input on the left
with left_col:
    st.header("Enter Patient Characteristics")

    # User input for various features
    gender = st.selectbox('Gender', ('Male', 'Female'))
    marital_status = st.selectbox('Marital Status', ('Married', 'Single'))
    age = st.number_input('Age', min_value=0, max_value=120, value=65, step=1)
    diabetes = st.selectbox('Diabetes', ('Yes', 'No'))
    hypertension = st.selectbox('Hypertension', ('Yes', 'No'))
    cesd_10 = st.slider('CESD-10 Score', 0, 30, 10)
    disability = st.selectbox('Disability', ('Yes', 'No'))
    sleep = st.slider('Sleep Duration (hours)', 0, 24, 7)  # Sleep duration
    cognitive_function = st.slider('Cognitive Function Score', 0, 21, 10)  # Cognitive function score
    adl = st.slider('ADL (6 items)', 0, 6, 3)  # ADL with 6 items
    life_satisfaction = st.slider('Life Satisfaction', 1, 5, 3)

    # Combine user input into a DataFrame
    input_data = pd.DataFrame({
        'Gender': [1 if gender == 'Male' else 0],
        'Marital Status': [1 if marital_status == 'Married' else 0],
        'Age': [age],
        'Diabetes': [1 if diabetes == 'Yes' else 0],
        'Hypertension': [1 if hypertension == 'Yes' else 0],
        'CESD-10': [cesd_10],
        'Disability': [1 if disability == 'Yes' else 0],
        'Sleep': [sleep],
        'Cognitive Function': [cognitive_function],
        'ADL': [adl],
        'Life Satisfaction': [life_satisfaction],
    })

# Results and feature importance on the right
with right_col:
    if st.button("Predict Frailty Probability"):
        # Predict the probability
        prediction_proba = clf.predict_proba(input_data)[:, 1] * 100  # Prediction probability in percentage
        predicted_class = clf.predict(input_data)

        # Generate advice based on the prediction result
        if predicted_class[0] == 1:
            advice = (
                f"According to the model, you may be at a higher risk of frailty, with an estimated probability of {prediction_proba[0]:.1f}%. "
                "It is recommended to closely monitor your health, particularly any changes in physical activity or quality of life. "
                "For a more accurate assessment, consider consulting a healthcare professional for further diagnosis and care recommendations."
            )
        else:
            advice = (
                f"The model predicts that your current risk of frailty is low, with an estimated probability of {100 - prediction_proba[0]:.1f}%. "
                "Although the risk is low, maintaining a healthy lifestyle remains crucial. "
                "Regular health check-ups are advised to ensure continuous monitoring of your health status."
            )

        st.write(advice)

        # Get feature importance
        feature_importances = clf.feature_importances_
        feature_names = input_data.columns

        # Visualize feature importance
        st.header("Feature Importance")
        plt.figure(figsize=(10, 6))  # Making the figure more compact
        plt.barh(feature_names, feature_importances, color='skyblue')
        plt.xlabel('Importance Score')
        plt.title('Feature Importance Ranking')
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()
