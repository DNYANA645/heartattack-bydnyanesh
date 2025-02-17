import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model and scaler
def load_model_and_scaler():
    try:
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return rf_model, scaler
    except FileNotFoundError as e:
        st.error(f"Error: Model or scaler file not found. Please ensure 'random_forest_model.pkl' and 'scaler.pkl' are in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

# Function for heart attack risk prediction
def predict_heart_attack_risk(user_input, scaler, model):
    """Predict heart attack risk and provide detailed advice."""

    # Scale the user input
    try:
        scaled_input = scaler.transform(np.array(user_input).reshape(1, -1))
    except Exception as e:
        st.error(f"Error scaling input data: {e}. Please check your input values.")
        return None, None, None, None, None, None, None  # Return None values to prevent further errors

    # Make prediction
    try:
        prediction = model.predict(scaled_input)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}. Please ensure the model is compatible with the input data.")
        return None, None, None, None, None, None, None  # Return None values to prevent further errors

    # Define messages based on prediction
    if prediction == 1:
        risk_level = "High"
        message = (
            "You are at high risk of a heart attack. Immediate consultation with a cardiologist is strongly advised. "
            "It is crucial to take this seriously and act promptly to mitigate potential health risks. 🚨"
        )

        precautions = [
            "1. Consult a Cardiologist Immediately: Schedule an appointment for a thorough evaluation. 👩‍⚕",
            "2. Modify Diet: Switch to a heart-healthy diet, reducing saturated and trans fats 🍔 and increasing fiber 🥦.",
            "3. Start Light Exercise: If cleared by a doctor, begin with low-impact activities. 🚶‍♀",
            "4. Strictly Adhere to Medications: Take all prescribed medications exactly as directed. 💊",
            "5. Monitor Symptoms Closely: Keep a close watch on any chest pain 💔, shortness of breath 😮‍💨, or unusual fatigue 😴."
        ]

        guidance = [
            "1. Follow Medication Plan: Adhere to your prescribed medication schedule without alterations. ⏰",
            "2. Adopt a Balanced Lifestyle: Focus on diet 🥗, stress management 🧘‍♀, and regular moderate activity 🚴.",
            "3. Schedule Regular Check-ups: Frequent check-ups help in monitoring your condition effectively. 📅",
            "4. Manage Stress Levels: Employ techniques like meditation 🧘 or yoga to lower stress. 😌",
            "5. Involve a Support System: Engage family and friends for emotional and practical support. 🫂"
        ]

        exercise = [
            "1. Consult Your Doctor: Get a tailored exercise plan from your healthcare provider. 👨‍⚕",
            "2. Start Slowly: Begin with very gentle activities such as walking 🚶 or light stretching.",
            "3. Build Gradually: Incrementally increase exercise intensity and duration over time. 📈",
            "4. Choose Enjoyable Activities: Opt for exercises you find pleasurable and motivating. 😄",
            "5. Listen To Your Body: Do not ignore any pain or discomfort; adjust accordingly. 🙏"
        ]

        diet = [
            "1. Prioritize Heart-Healthy Foods: Emphasize fruits 🍎, vegetables 🥦, lean proteins 🍗, and whole grains 🌾.",
            "2. Limit Unhealthy Fats: Minimize saturated and trans fats to protect your arteries. 🍟",
            "3. Reduce Sodium Intake: Lower sodium to manage blood pressure effectively. 🧂",
            "4. Stay Hydrated: Drink plenty of water to support overall cardiovascular function. 💧",
            "5. Avoid Processed Foods: Reduce or eliminate processed foods high in sugars and fats. 🍩"
        ]

        medications = [
            "1. Stick to Prescriptions: Strictly follow prescribed medication dosages and timings. 💊",
            "2. Understand Each Medication: Know the purpose and potential side effects of each medicine. ℹ",
            "3. Regular Review: Review all medications with your healthcare provider regularly. 👩‍⚕",
            "4. Do Not Self-Medicate: Avoid taking any other medications without consulting your doctor. 🚫",
            "5. Report Side Effects: Promptly report any side effects to your doctor. 🗣"
        ]

    else:
        risk_level = "Low"
        message = (
            "Your heart attack risk appears to be low. ✅ However, it is essential to maintain a healthy lifestyle 🏃‍♀ to ensure long-term cardiovascular health. ❤️"
        )

        precautions = [
            "1. Continue Regular Check-ups: Maintain routine appointments with your primary care physician. 👩‍⚕",
            "2. Monitor Health Metrics: Keep track of blood pressure, cholesterol, and blood sugar levels. 📊",
            "3. Maintain a Healthy Lifestyle: Continue with a balanced diet 🥗 and regular exercise routine 🏋‍♀.",
            "4. Stay Informed: Be proactive in learning about heart health and risk factors. 📚",
            "5. Plan for Emergencies: Have a plan in place in case of any sudden health issues. 📅"
        ]

        guidance = [
            "1. Maintain Balanced Diet: Ensure a diverse intake of fruits 🍎, vegetables 🥦, and whole grains 🌾.",
            "2. Stay Active Daily: Engage in at least 150 minutes of moderate aerobic exercise per week. 🏃",
            "3. Practice Stress Reduction: Use techniques like mindfulness 🧘 or yoga to manage stress. 😌",
            "4. Moderate Alcohol Intake: Adhere to recommended limits for alcohol consumption. 🍺",
            "5. Avoid Smoking: Refrain from smoking 🚭 to maintain optimal heart health. ❤️"
        ]

        exercise = [
            "1. Mix Up Activities: Include cardio 🏃, strength training 💪, and flexibility exercises.",
            "2. Be Consistent: Make physical activity a regular part of your daily routine. 📅",
            "3. Enjoy Your Workouts: Select activities that you find enjoyable and motivating. 😄",
            "4. Set Achievable Goals: Establish realistic fitness goals tailored to your ability. 🎯",
            "5. Listen to Your Body's Signals: Adjust your activity level based on how you feel. 🙏"
        ]

        diet = [
            "1. Focus on Whole Foods: Limit processed foods, emphasizing fruits 🍎, vegetables 🥦, and whole grains 🌾.",
            "2. Stay Hydrated: Drink plenty of water throughout the day. 💧",
            "3. Control Portions: Practice mindful eating to maintain a healthy weight. ⚖",
            "4. Plan Your Meals: Prepare meals in advance to make healthier choices. 🍱",
            "5. Seek Nutritional Advice: Consult a nutritionist for personalized guidance if needed. 👩‍⚕"
        ]

        medications = [
            "1. Consult Your Doctor Regularly: Discuss any health concerns or medication questions with them. 👩‍⚕",
            "2. Prioritize Prevention: Focus on lifestyle changes that can prevent heart issues. 💪",
            "3. Review Annually: Review your medications and health status annually with your doctor. 📅",
            "4. Stay Aware of Changes: Note any changes in how you feel after starting or stopping medications and report them promptly. 📝",
            "5. Be Informed: Educate yourself about your health conditions and medications. 📚"
        ]

    return risk_level, message, precautions, guidance, exercise, diet, medications

# Set up Streamlit app
st.title("Heart Attack Risk Prediction-❤️")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page:", ["Home", "About App & Modules", "Symptoms Information"])

if page == "Home":
    

    # Load model and scaler
    rf_model, scaler = load_model_and_scaler()

    # Check if model and scaler loaded successfully
    if not rf_model or not scaler:
        st.error("Model and scaler could not be loaded.")
        st.stop()

    # Collect input features from the user through Streamlit widgets
    st.sidebar.header("Patient Information")

    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)

    # Define the options for the selectbox
    sex_options = {"Female": 0, "Male": 1}
    sex_label = st.sidebar.selectbox("Select Sex", options=list(sex_options.keys()))
    sex = sex_options[sex_label]  # Set the value based on selection

    cp = st.sidebar.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=90, max_value=250, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.sidebar.selectbox("Resting ECG (restecg)", options=[0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate Achieved (thalach)", min_value=60,max_value=220,value=150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina (exang)", options=[0 , 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0., max_value=10., value=0.)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0 , 1 , 2])
    ca = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy (ca)", min_value=0 , max_value=4 , value=0)
    thal = st.sidebar.selectbox("Thalassemia (thal)", options=[0 , 1 , 2 , 3])

    # Create a button to trigger the prediction
    if st.sidebar.button("Predict"):
        # Prepare the user input for prediction
        user_input = [age , sex , cp , trestbps , chol , fbs , restecg ,
                      thalach , exang , oldpeak ,
                      slope ,
                      ca ,
                      thal]

        # Call the prediction function
        risk_level , message , precautions , guidance , exercise , diet , medications = predict_heart_attack_risk(user_input , scaler , rf_model)

        # Check if the prediction was successful
        if risk_level is not None:
            # Display the prediction results
             # Correcting the Heading Here
            st.subheader("Prediction Results")

            # Using Markdown to make Risk Level more attractive
            if risk_level == "High":
                st.markdown(f"<h3 style='color:red;'>Risk Level: <span style='font-weight: bold; color: red;'>{risk_level}</span></h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:green;'>Risk Level: <span style='font-weight: bold; color: green;'>{risk_level}</span></h3>", unsafe_allow_html=True)

            st.write(message)

            st.subheader("RECOMMENDATIONS:")

            st.markdown("PRECAUTIONS-")
            for rec in precautions:
                st.write(rec)

            st.markdown("---")

            st.markdown("GUIDANCE-")
            for rec in guidance:
                st.write(rec)

            st.markdown("---")

            st.markdown("EXERCISE-")
            for rec in exercise:
                st.write(rec)

            st.markdown("---")

            st.markdown("DIET-")
            for rec in diet:
                st.write(rec)

            st.markdown("---")

            st.markdown("MEDICATIONS")
            for rec in medications:
                st.write(rec)

elif page == "About App & Modules":
    # About App Section
    st.header("About This Application")
    st.write("""
    This application is designed to assess an individual's risk of having a heart attack based on various health parameters.
    It utilizes machine learning algorithms to analyze user inputs such as age,
    sex, blood pressure levels,
    cholesterol levels,
    chest pain type,
    maximum heart rate achieved,
    fasting blood sugar levels,
    resting ECG results,
    number of major vessels colored by fluoroscopy,
    slope of peak exercise ST segment,
    thalassemia status,
    etc.
    """)

    # About Modules Section
    st.subheader("Modules Used")
    st.write("""
    - Risk Prediction Module: This module employs a Random Forest classifier trained on historical patient data.
      It predicts whether an individual is at high or low risk for heart attacks based on their input parameters.

    - Recommendation Module: After assessing risk levels,
      this module provides tailored recommendations regarding lifestyle changes,
      dietary adjustments,
      medication adherence,
      exercise plans,
      etc., aimed at improving cardiovascular health.

    - User Interface Module: This application features an intuitive user interface built using Streamlit,
      allowing users easy navigation through different sections including predictions,
      recommendations,
      educational content about symptoms,
      etc.

    - Data Handling Module: This module manages data preprocessing steps such as scaling inputs before feeding them into the model.
      It ensures that all inputs are standardized according to what the model expects.

    - Visualization Module: Although not implemented yet,
      this module can be used in future versions for visualizing user data trends over time or comparing different metrics.
    """)

elif page == "Symptoms Information":
    # Symptoms Information Section
    st.header("Heart Attack Symptoms")
    st.write("""
    Recognizing the symptoms of a heart attack is crucial for timely intervention.

    Common symptoms include:

    - Chest Pain or Discomfort: Often described as pressure or squeezing sensation.
    
    - Shortness of Breath: May occur with or without chest discomfort.
    
    - Pain or Discomfort in Other Areas: Such as arms (especially left arm), shoulder blades,
      neck jaw back stomach.
    
    - Nausea/Vomiting: Some individuals may experience stomach upset along with other symptoms.
    
    - Lightheadedness/Fainting: Feeling dizzy or faint can also indicate an issue related to heart health.
    
    If you experience any of these symptoms especially if they last more than few minutes seek immediate medical attention!
    
    Remember that symptoms can vary between individuals especially between men & women!
    """)

# Add a disclaimer at the bottom of each page
st.markdown(
    "<hr style='border:2px solid gray'>", unsafe_allow_html=True)  # Adding horizontal line before disclaimer
st.markdown(
    "<strong>Disclaimer:</strong> This app provides general predictions based on user input but should not be used as substitute for professional medical advice."
    + "<br>Consult with qualified healthcare provider regarding any health concerns.", unsafe_allow_html=True)