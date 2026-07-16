import streamlit as st
import numpy as np
import joblib
import pandas as pd
import time

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Load Model
model = joblib.load("SVM_heart.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
# ---------------- CSS ---------------- #

st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
background-size:400% 400%;
animation: gradient 15s ease infinite;
color:white;
}

@keyframes gradient{
0%{background-position:0% 50%;}
50%{background-position:100% 50%;}
100%{background-position:0% 50%;}
}

.main-title{
font-size:55px;
font-weight:900;
text-align:center;
color:white;
animation: heartbeat 1.2s infinite;
}

@keyframes heartbeat{
0%{transform:scale(1);}
25%{transform:scale(1.08);}
40%{transform:scale(0.95);}
60%{transform:scale(1.08);}
100%{transform:scale(1);}
}

.subtitle{
text-align:center;
font-size:20px;
color:#dddddd;
margin-bottom:30px;
}

.card{
background:rgba(255,255,255,0.12);
padding:20px;
border-radius:20px;
backdrop-filter: blur(20px);
box-shadow:0 8px 32px rgba(0,0,0,0.35);
transition:0.4s;
margin-bottom:20px;
}

.card:hover{
transform:translateY(-5px);
box-shadow:0 15px 35px rgba(0,0,0,0.5);
}

div.stButton>button{
width:100%;
height:60px;
border-radius:15px;
font-size:22px;
font-weight:bold;
background:linear-gradient(45deg,#ff416c,#ff4b2b);
color:white;
border:none;
transition:0.3s;
}

div.stButton>button:hover{
transform:scale(1.05);
box-shadow:0px 0px 20px red;
}

.metric{
background:rgba(255,255,255,0.1);
padding:15px;
border-radius:15px;
text-align:center;
}

hr{
border:1px solid rgba(255,255,255,0.2);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
"""
<div class="main-title">
❤️ Heart Disease Prediction System
</div>

<div class="subtitle">
AI Powered Healthcare Assistant
</div>
""",
unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966488.png", width=120)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
"",
[
"Prediction",
"About Model",
"Patient History"
]
)

if page=="Prediction":

    st.markdown('<div class="card">',unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)

    with col1:

        age=st.slider("Age",20,100,45)

        sex=st.selectbox("Gender",["Male","Female"])

        cp=st.selectbox("Chest Pain Type",[0,1,2,3])

        trestbps=st.number_input("Resting Blood Pressure",80,220,120)

        chol=st.number_input("Cholesterol",100,600,240)

    with col2:

        fbs=st.selectbox("Fasting Blood Sugar",[0,1])

        restecg=st.selectbox("Rest ECG",[0,1,2])

        thalach=st.slider("Maximum Heart Rate",60,220,150)

        exang=st.selectbox("Exercise Induced Angina",[0,1])

        oldpeak=st.slider("Old Peak",0.0,6.5,1.0)

    with col3:

        slope=st.selectbox("Slope",[0,1,2])

        ca=st.selectbox("Number of Major Vessels",[0,1,2,3,4])

        thal=st.selectbox("Thal",[0,1,2,3])

        weight=st.number_input("Weight (kg)",30,200,70)

        height=st.number_input("Height (cm)",100,230,170)

    st.markdown("</div>",unsafe_allow_html=True)

    bmi=weight/((height/100)**2)

    st.markdown("### 📊 Patient Summary")

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Age",age)

    c2.metric("BMI",round(bmi,2))

    c3.metric("Cholesterol",chol)

    c4.metric("Heart Rate",thalach)

    st.write("")

    if st.button("❤️ Predict Heart Disease"):

      progress = st.progress(0)

for i in range(100):
    time.sleep(0.01)
    progress.progress(i + 1)

# Convert gender
sex_value = 1 if sex == "Male" else 0

# Create input
input_data = np.array([[
    age,
    sex_value,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_scaled)[0]

# Probability
probability = model.predict_proba(input_scaled)[0][1]

st.success("Prediction Completed Successfully!")

if prediction == 1:

    st.error("❤️ High Risk of Heart Disease")

    st.metric(
        "Risk Probability",
        f"{probability*100:.2f}%"
    )

    st.progress(int(probability * 100))

    color = "#ff4b4b"
    result = "HIGH RISK"

else:

    st.success("💚 Low Risk of Heart Disease")

    st.metric(
        "Safety Probability",
        f"{(1-probability)*100:.2f}%"
    )

    st.progress(int((1-probability) * 100))

    color = "#00C853"
    result = "LOW RISK"

st.markdown(
f"""
<div style='
background:{color};
padding:25px;
border-radius:20px;
text-align:center;
font-size:35px;
font-weight:bold;
color:white;
margin-top:20px;
'>

Prediction : {result}

</div>
""",
unsafe_allow_html=True
)

df = pd.DataFrame({
    "Feature": [
        "Age",
        "Gender",
        "Chest Pain",
        "Blood Pressure",
        "Cholesterol",
        "Heart Rate",
        "Risk Probability"
    ],
    "Value": [
        age,
        sex,
        cp,
        trestbps,
        chol,
        thalach,
        f"{probability*100:.2f}%"
    ]
})

st.dataframe(df, use_container_width=True)

st.markdown("<hr>",unsafe_allow_html=True)

st.markdown(
"""
<center>

Made with ❤️ using Streamlit

</center>
""",
unsafe_allow_html=True
)