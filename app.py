#Building the streamlit app

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ──
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #1a1a2e;
            color: #eaeaea;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #16213e;
            color: white;
        }

        /* Sidebar text */
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stNumberInput label {
            color: #DAA520 !important;
            font-weight: 600;
        }

        /* Header banner */
        .banner {
            background: linear-gradient(135deg, #8B0000, #D2691E, #DAA520);
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 25px;
        }
        .banner h1 {
            color: white;
            font-size: 2.2em;
            margin: 0;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        .banner p {
            color: #ffe8c0;
            font-size: 1em;
            margin-top: 8px;
        }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #16213e, #0f3460);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid #DAA520;
            margin: 5px;
        }
        .metric-card h3 {
            color: #DAA520;
            font-size: 0.9em;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-card h2 {
            color: white;
            font-size: 1.8em;
            margin: 0;
            font-weight: 800;
        }

        /* Result cards */
        .result-graduate {
            background: linear-gradient(135deg, #1a4a1a, #2d7a2d);
            border: 2px solid #4CAF50;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        }
        .result-enrolled {
            background: linear-gradient(135deg, #4a3000, #7a5200);
            border: 2px solid #DAA520;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        }
        .result-dropout {
            background: linear-gradient(135deg, #4a0000, #8B0000);
            border: 2px solid #FF4444;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        }
        .result-card h1 {
            font-size: 2.5em;
            margin: 0;
            color: white;
        }
        .result-card p {
            color: #eaeaea;
            font-size: 1em;
            margin-top: 10px;
        }

        /* Section headers */
        .section-header {
            color: #DAA520;
            font-size: 1.3em;
            font-weight: 700;
            border-bottom: 2px solid #D2691E;
            padding-bottom: 8px;
            margin-bottom: 15px;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #888;
            font-size: 0.85em;
            padding: 20px;
            border-top: 1px solid #333;
            margin-top: 40px;
        }

        /* Predict button */
        .stButton > button {
            background: linear-gradient(135deg, #8B0000, #DAA520);
            color: white;
            font-weight: 700;
            font-size: 1.1em;
            border: none;
            border-radius: 10px;
            padding: 12px 30px;
            width: 100%;
            cursor: pointer;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #DAA520, #8B0000);
            transform: scale(1.02);
        }

        /* Chart backgrounds */
        .stPlotlyChart, .stImage {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load saved files ──
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')
best_model_name = joblib.load('best_model_name.pkl')

# ── Banner ──
st.markdown(f"""
    <div class="banner">
        <h1>🎓 Student Academic Performance Predictor</h1>
        <p>Powered by {best_model_name} &nbsp;|&nbsp;
        Amaraegbu Divine &nbsp;|&nbsp;
        Department of Computer Science</p>
    </div>
""", unsafe_allow_html=True)

# ── Sidebar ──
st.sidebar.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <h2 style='color:#DAA520; font-size:1.2em;'>📋 Student Details</h2>
        <p style='color:#aaa; font-size:0.8em;'>Fill in all fields and click Predict</p>
    </div>
""", unsafe_allow_html=True)

def user_input():
    st.sidebar.markdown("**🎓 Academic Information**")
    prev_qual = st.sidebar.number_input("Previous Qualification", 1, 40, 1)
    prev_qual_grade = st.sidebar.number_input("Previous Qualification Grade", 0.0, 200.0, 120.0)
    admission_grade = st.sidebar.number_input("Admission Grade", 0.0, 200.0, 127.0)
    course = st.sidebar.number_input("Course Code", 1, 9999, 171)
    application_mode = st.sidebar.number_input("Application Mode", 1, 57, 1)
    application_order = st.sidebar.number_input("Application Order", 0, 9, 1)
    attendance = st.sidebar.selectbox("Daytime/Evening Attendance", [1, 0],
                                       format_func=lambda x: "Daytime" if x == 1 else "Evening")

    st.sidebar.markdown("**📊 1st Semester Performance**")
    cu1_credited = st.sidebar.number_input("Units Credited", 0, 20, 0, key="cu1c")
    cu1_enrolled = st.sidebar.number_input("Units Enrolled", 0, 26, 6, key="cu1e")
    cu1_evals = st.sidebar.number_input("Units Evaluated", 0, 45, 6, key="cu1ev")
    cu1_approved = st.sidebar.number_input("Units Approved", 0, 26, 5, key="cu1a")
    cu1_grade = st.sidebar.number_input("Average Grade", 0.0, 20.0, 12.0, key="cu1g")
    cu1_no_eval = st.sidebar.number_input("Without Evaluations", 0, 12, 0, key="cu1n")

    st.sidebar.markdown("**📊 2nd Semester Performance**")
    cu2_credited = st.sidebar.number_input("Units Credited", 0, 20, 0, key="cu2c")
    cu2_enrolled = st.sidebar.number_input("Units Enrolled", 0, 23, 6, key="cu2e")
    cu2_evals = st.sidebar.number_input("Units Evaluated", 0, 45, 6, key="cu2ev")
    cu2_approved = st.sidebar.number_input("Units Approved", 0, 20, 5, key="cu2a")
    cu2_grade = st.sidebar.number_input("Average Grade", 0.0, 20.0, 12.0, key="cu2g")
    cu2_no_eval = st.sidebar.number_input("Without Evaluations", 0, 12, 0, key="cu2n")

    st.sidebar.markdown("**👤 Demographic Information**")
    age = st.sidebar.number_input("Age at Enrollment", 17, 70, 20)
    gender = st.sidebar.selectbox("Gender", [1, 0],
                                   format_func=lambda x: "Male" if x == 1 else "Female")
    marital_status = st.sidebar.selectbox("Marital Status", [1,2,3,4,5,6],
                                           format_func=lambda x: {
                                               1:"Single", 2:"Married",
                                               3:"Widower", 4:"Divorced",
                                               5:"Facto Union", 6:"Legally Separated"}[x])
    nationality = st.sidebar.number_input("Nationality Code", 1, 109, 1)
    international = st.sidebar.selectbox("International Student", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")
    displaced = st.sidebar.selectbox("Displaced", [0, 1],
                                      format_func=lambda x: "No" if x == 0 else "Yes")
    special_needs = st.sidebar.selectbox("Educational Special Needs", [0, 1],
                                          format_func=lambda x: "No" if x == 0 else "Yes")

    st.sidebar.markdown("**💰 Socioeconomic Information**")
    tuition = st.sidebar.selectbox("Tuition Fees Up to Date", [1, 0],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
    scholarship = st.sidebar.selectbox("Scholarship Holder", [0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Yes")
    debtor = st.sidebar.selectbox("Debtor", [0, 1],
                                   format_func=lambda x: "No" if x == 0 else "Yes")
    mothers_qual = st.sidebar.number_input("Mother's Qualification", 1, 44, 19)
    fathers_qual = st.sidebar.number_input("Father's Qualification", 1, 44, 12)
    mothers_occ = st.sidebar.number_input("Mother's Occupation", 0, 194, 5)
    fathers_occ = st.sidebar.number_input("Father's Occupation", 0, 194, 9)

    st.sidebar.markdown("**🌍 Macroeconomic Indicators**")
    unemployment = st.sidebar.number_input("Unemployment Rate (%)", 0.0, 25.0, 10.8)
    inflation = st.sidebar.number_input("Inflation Rate (%)", -1.0, 5.0, 1.4)
    gdp = st.sidebar.number_input("GDP", -5.0, 5.0, 1.74)

    data = [marital_status, application_mode, application_order, course,
            attendance, prev_qual, prev_qual_grade, nationality,
            mothers_qual, fathers_qual, mothers_occ, fathers_occ,
            admission_grade, displaced, special_needs, debtor,
            tuition, gender, scholarship, age, international,
            cu1_credited, cu1_enrolled, cu1_evals, cu1_approved,
            cu1_grade, cu1_no_eval, cu2_credited, cu2_enrolled,
            cu2_evals, cu2_approved, cu2_grade, cu2_no_eval,
            unemployment, inflation, gdp]

    return np.array(data).reshape(1, -1)

input_data = user_input()

# ── Predict button ──
predict_clicked = st.sidebar.button("🔍 Predict Outcome")

# ── Main content ──
if predict_clicked:
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
    predicted_class = le.inverse_transform(prediction)[0]

    # Result card
    st.markdown('<p class="section-header">📌 Prediction Result</p>',
                unsafe_allow_html=True)

    if predicted_class == "Graduate":
        st.markdown(f"""
            <div class="result-graduate result-card">
                <h1>✅ GRADUATE</h1>
                <p>This student is predicted to successfully complete their academic program.</p>
            </div>
        """, unsafe_allow_html=True)

    elif predicted_class == "Enrolled":
        st.markdown(f"""
            <div class="result-enrolled result-card">
                <h1>⚠️ ENROLLED</h1>
                <p>This student is predicted to remain enrolled but has not yet graduated.
                Academic support is recommended.</p>
            </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
            <div class="result-dropout result-card">
                <h1>🚨 DROPOUT</h1>
                <p>This student is at high risk of dropping out.
                Immediate intervention is strongly recommended.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability metric cards
    st.markdown('<p class="section-header">📊 Prediction Probabilities</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    class_probs = dict(zip(le.classes_, probabilities))

    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>🚨 Dropout</h3>
                <h2>{class_probs.get('Dropout', 0):.1%}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <h3>⚠️ Enrolled</h3>
                <h2>{class_probs.get('Enrolled', 0):.1%}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <h3>✅ Graduate</h3>
                <h2>{class_probs.get('Graduate', 0):.1%}</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Probability bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    colors = ['#FF4444', '#DAA520', '#4CAF50']
    bars = ax.bar(le.classes_, probabilities, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', color='white')
    ax.set_title('Class Probability Distribution', color='#DAA520', fontweight='bold')
    ax.tick_params(colors='white')
    ax.bar_label(bars, fmt='%.3f', padding=3, color='white', fontweight='bold')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    st.pyplot(fig)

else:
    st.markdown("""
        <div style='text-align:center; padding: 60px 20px;
                    background: #16213e; border-radius: 15px;
                    border: 1px dashed #DAA520;'>
            <h2 style='color:#DAA520;'>👈 Fill in Student Details</h2>
            <p style='color:#aaa;'>Use the sidebar to enter student information,
            then click Predict Outcome to get a prediction.</p>
        </div>
    """, unsafe_allow_html=True)

# ── Visualizations ──
import os
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">📈 Model Performance & Insights</p>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if os.path.exists('model_comparison.png'):
        st.image('model_comparison.png',
                 caption='Model Performance Comparison', use_column_width=True)
    else:
        st.warning("Run Step 5 to generate model_comparison.png")

with col2:
    if os.path.exists('confusion_matrix.png'):
        st.image('confusion_matrix.png',
                 caption='Confusion Matrix', use_column_width=True)
    else:
        st.warning("Run Step 5 to generate confusion_matrix.png")

if os.path.exists('feature_importance.png'):
    st.image('feature_importance.png',
             caption='Top 15 Feature Importances', use_column_width=True)
else:
    st.warning("Run Step 5 to generate feature_importance.png")

# ── Footer ──
st.markdown("""
    <div class="footer">
        Student Academic Performance Prediction System &nbsp;|&nbsp;
        Amaraegbu Divine &nbsp;|&nbsp;
        Department of Computer Science &nbsp;|&nbsp; 2025
    </div>
""", unsafe_allow_html=True)
