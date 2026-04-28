#Building the streamlit app
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ── Page config ──
st.set_page_config(
    page_title="Student Academic Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
    <style>
        .stApp { background-color: #1a1a2e; color: #eaeaea; }
        [data-testid="stSidebar"] { background-color: #16213e; color: white; }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stNumberInput label {
            color: #DAA520 !important; font-weight: 600;
        }
        .banner {
            background: linear-gradient(135deg, #8B0000, #D2691E, #DAA520);
            padding: 30px; border-radius: 15px;
            text-align: center; margin-bottom: 25px;
        }
        .banner h1 { color: white; font-size: 2.2em; margin: 0;
            font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
        .banner p { color: #ffe8c0; font-size: 1em; margin-top: 8px; }
        .metric-card {
            background: linear-gradient(135deg, #16213e, #0f3460);
            border-radius: 12px; padding: 20px; text-align: center;
            border: 1px solid #DAA520; margin: 5px;
        }
        .metric-card h3 { color: #DAA520; font-size: 0.9em;
            margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
        .metric-card h2 { color: white; font-size: 1.8em; margin: 0; font-weight: 800; }
        .result-graduate {
            background: linear-gradient(135deg, #1a4a1a, #2d7a2d);
            border: 2px solid #4CAF50; border-radius: 15px; padding: 25px; text-align: center;
        }
        .result-enrolled {
            background: linear-gradient(135deg, #4a3000, #7a5200);
            border: 2px solid #DAA520; border-radius: 15px; padding: 25px; text-align: center;
        }
        .result-dropout {
            background: linear-gradient(135deg, #4a0000, #8B0000);
            border: 2px solid #FF4444; border-radius: 15px; padding: 25px; text-align: center;
        }
        .result-card h1 { font-size: 2.5em; margin: 0; color: white; }
        .result-card p { color: #eaeaea; font-size: 1em; margin-top: 10px; }
        .section-header {
            color: #DAA520; font-size: 1.3em; font-weight: 700;
            border-bottom: 2px solid #D2691E; padding-bottom: 8px; margin-bottom: 15px;
        }
        .footer {
            text-align: center; color: #888; font-size: 0.85em;
            padding: 20px; border-top: 1px solid #333; margin-top: 40px;
        }
        .stButton > button {
            background: linear-gradient(135deg, #8B0000, #DAA520);
            color: white; font-weight: 700; font-size: 1.1em;
            border: none; border-radius: 10px;
            padding: 12px 30px; width: 100%; cursor: pointer;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #DAA520, #8B0000);
        }
    </style>
""", unsafe_allow_html=True)

# ── Load saved files ──
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
feature_names = joblib.load('feature_names.pkl')
best_model_name = joblib.load('best_model_name.pkl')

# ══════════════════════════════════════════
# DROPDOWN MAPPINGS
# ══════════════════════════════════════════

marital_map = {
    'Single': 1, 'Married': 2, 'Widower': 3,
    'Divorced': 4, 'Facto Union': 5, 'Legally Separated': 6
}

application_mode_map = {
    '1st phase - General quota': 1,
    'Ordinance No. 612/93': 2,
    '1st phase - Special quota (Madeira Island)': 5,
    'Holders of other higher courses': 7,
    'Ordinance No. 854-B/99': 10,
    'International student (over 23 years)': 15,
    '1st phase - Special quota (Azores Island)': 16,
    '2nd phase - General quota': 17,
    '3rd phase - General quota': 18,
    'Holders of other higher courses (1st cycle)': 26,
    'Over 23 years old': 27,
    'Over 23 years old (special quota)': 39,
    'Transfer': 42,
    'Change of course': 43,
    'Technological specialization diploma holders': 44,
    'Change of institution/course': 51,
    'Short cycle diploma holders': 53,
    'Change of institution/course (International)': 57,
}

course_map = {
    'Biofuel Production Technologies': 33,
    'Animation and Multimedia Design': 171,
    'Social Service (evening attendance)': 8014,
    'Agronomy': 9003,
    'Communication Design': 9070,
    'Veterinary Nursing': 9085,
    'Informatics Engineering (closest to Computer Science)': 9119,
    'Equiniculture': 9130,
    'Management': 9147,
    'Social Service': 9238,
    'Tourism': 9254,
    'Nursing': 9500,
    'Oral Hygiene': 9556,
    'Management (evening attendance)': 9670,
    'Journalism and Communication': 9773,
    'Basic Education': 9853,
    'Advertising and Marketing Management': 9991,
}

prev_qual_map = {
    'Secondary education': 1,
    "Higher education - Bachelor's degree": 2,
    'Higher education - Degree': 3,
    "Higher education - Master's": 4,
    'Higher education - Doctorate': 5,
    'Frequency of higher education': 6,
    '12th year of schooling - not completed': 9,
    '11th year of schooling - not completed': 10,
    'Other - 11th year of schooling': 12,
    '10th year of schooling': 14,
    '10th year of schooling - not completed': 15,
    'Basic education 3rd cycle (9th/10th/11th year) or equiv.': 19,
    'Basic education 2nd cycle (6th/7th/8th year) or equiv.': 38,
    'Technological specialization course': 39,
    'Higher education - degree (1st cycle)': 40,
    'Professional higher technical course': 42,
    'Higher education - master (2nd cycle)': 43,
}

nationality_map = {
    'Portuguese': 1, 'German': 2, 'Spanish': 6, 'Italian': 11, 'Nigerian': 12,
    'Dutch': 13, 'English': 14, 'Lithuanian': 17, 'Angolan': 21,
    'Cape Verdean': 22, 'Guinean': 24, 'Mozambican': 25,
    'Santomean': 26, 'Turkish': 32, 'Brazilian': 41,
    'Romanian': 62, 'Moldova (Republic of)': 100, 'Mexican': 101,
    'Ukrainian': 103, 'Russian': 105, 'Cuban': 108, 'Colombian': 109,
}

qualification_map = {
    'Secondary Education - 12th Year of Schooling or Eq.': 1,
    "Higher Education - Bachelor's Degree": 2,
    'Higher Education - Degree': 3,
    "Higher Education - Master's": 4,
    'Higher Education - Doctorate': 5,
    'Frequency of Higher Education': 6,
    '12th Year of Schooling - Not Completed': 9,
    '11th Year of Schooling - Not Completed': 10,
    '7th Year (Old)': 11,
    'Other - 11th Year of Schooling': 12,
    '10th Year of Schooling': 14,
    'General commerce course': 18,
    'Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.': 19,
    'Technical-professional course': 22,
    '7th year of schooling': 26,
    '2nd cycle of the general high school course': 27,
    '9th Year of Schooling - Not Completed': 29,
    '8th year of schooling': 30,
    'Unknown': 34,
    "Can't read or write": 35,
    'Can read without having a course': 36,
    'Basic education 1st cycle (4th/5th year) or equiv.': 37,
    'Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.': 38,
    'Technological specialization course': 39,
    'Higher education - degree (1st cycle)': 40,
    'Specialized higher studies course': 41,
    'Professional higher technical course': 42,
    'Higher Education - Master (2nd cycle)': 43,
    'Higher Education - Doctorate (3rd cycle)': 44,
}

occupation_map = {
    'Student': 0,
    'Representatives of the Legislative Power and Executive Bodies, Directors and Executive Managers': 1,
    'Specialists in Intellectual and Scientific Activities': 2,
    'Intermediate Level Technicians and Professions': 3,
    'Administrative staff': 4,
    'Personal Services, Security and Safety Workers and Sellers': 5,
    'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry': 6,
    'Skilled Workers in Industry, Construction and Craftsmen': 7,
    'Equipment and Machines Operators and Assembly Workers': 8,
    'Unskilled Workers': 9,
    'Armed Forces Professions': 10,
    'Other Situation': 90,
    '(Blank)': 99,
    'Health professionals': 122,
    'Teachers': 123,
    'Specialists in information and communication technologies (ICT)': 125,
    'Intermediate level science and engineering technicians and associate professionals': 131,
    'Technicians and associate professionals in health': 132,
    'Intermediate level technicians from legal, social, sports, cultural and similar services': 134,
    'Office workers, secretaries in general and data processing operators': 141,
    'Data, accounting, statistical, financial services and registry-related support workers': 143,
    'Other support workers': 144,
    'Personal service workers': 151,
    'Sellers': 152,
    'Personal care workers and the like': 153,
    'Skilled construction workers and the like, except electricians': 171,
    'Skilled workers in printing, precision manufacturing, jewelry, craftsmen and the like': 173,
    'Workers in food processing, woodworking, clothing and other industries and crafts': 175,
    'Cleaning workers': 191,
    'Unskilled workers in agriculture, animal production, fisheries and forestry': 192,
    'Unskilled workers in extractive industry, construction, manufacturing and transport': 193,
    'Meal preparation assistants': 194,
}

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

    prev_qual_label = st.sidebar.selectbox(
        "Previous Qualification", options=list(prev_qual_map.keys()))
    prev_qual = prev_qual_map[prev_qual_label]

    prev_qual_grade = st.sidebar.number_input(
        "Previous Qualification Grade (max: 200)",
        min_value=0.0, max_value=200.0, value=120.0,
        help="Grade of previous qualification. Maximum value is 200.")

    admission_grade = st.sidebar.number_input(
        "Admission Grade (max: 200)",
        min_value=0.0, max_value=200.0, value=127.0,
        help="Admission grade at enrollment. Maximum value is 200.")

    course_label = st.sidebar.selectbox(
    "Course",
    options=list(course_map.keys()),
    index=list(course_map.keys()).index('Informatics Engineering (closest to Computer Science')
    )
    course = course_map[course_label]
    st.sidebar.caption("ℹ️ This dataset is sourced from Portuguese higher education institutions. Select the course closest to your student's field of study.")

    app_mode_label = st.sidebar.selectbox(
        "Application Mode", options=list(application_mode_map.keys()))
    application_mode = application_mode_map[app_mode_label]

    application_order = st.sidebar.number_input(
        "Application Order (0-9)", min_value=0, max_value=9, value=1,
        help="The order in which the student applied (between 0 and 9).")

    attendance = st.sidebar.selectbox(
        "Daytime/Evening Attendance", options=['Daytime', 'Evening'])
    attendance = 1 if attendance == 'Daytime' else 0

    st.sidebar.markdown("**📊 1st Semester Performance**")
    st.sidebar.info(
        "ℹ️ **1st Semester fields explained:**\n"
        "- **Units Credited:** Curricular units given credit (e.g. from prior learning)\n"
        "- **Units Enrolled:** Total units the student registered for\n"
        "- **Units Evaluated:** Units the student actually sat for assessment\n"
        "- **Units Approved:** Units the student successfully passed\n"
        "- **Average Grade:** Mean grade across all units (max: 20)\n"
        "- **Without Evaluations:** Units enrolled in but never assessed"
    )
    cu1_credited = st.sidebar.number_input("Units Credited", 0, 20, 0, key="cu1c")
    cu1_enrolled = st.sidebar.number_input("Units Enrolled", 0, 26, 6, key="cu1e")
    cu1_evals = st.sidebar.number_input("Units Evaluated", 0, 45, 6, key="cu1ev")
    cu1_approved = st.sidebar.number_input("Units Approved", 0, 26, 5, key="cu1a")
    cu1_grade = st.sidebar.number_input(
        "Average Grade (max: 20)", 0.0, 20.0, 12.0, key="cu1g",
        help="Average grade in the 1st semester. Maximum value is 20.")
    cu1_no_eval = st.sidebar.number_input("Without Evaluations", 0, 12, 0, key="cu1n")

    st.sidebar.markdown("**📊 2nd Semester Performance**")
    st.sidebar.info(
        "ℹ️ **2nd Semester fields explained:**\n"
        "- **Units Credited:** Curricular units given credit in 2nd semester\n"
        "- **Units Enrolled:** Total units registered for in 2nd semester\n"
        "- **Units Evaluated:** Units assessed in 2nd semester\n"
        "- **Units Approved:** Units successfully passed in 2nd semester\n"
        "- **Average Grade:** Mean grade across 2nd semester units (max: 20)\n"
        "- **Without Evaluations:** Units enrolled but never assessed in 2nd semester"
    )
    cu2_credited = st.sidebar.number_input("Units Credited", 0, 20, 0, key="cu2c")
    cu2_enrolled = st.sidebar.number_input("Units Enrolled", 0, 23, 6, key="cu2e")
    cu2_evals = st.sidebar.number_input("Units Evaluated", 0, 45, 6, key="cu2ev")
    cu2_approved = st.sidebar.number_input("Units Approved", 0, 20, 5, key="cu2a")
    cu2_grade = st.sidebar.number_input(
        "Average Grade (max: 20)", 0.0, 20.0, 12.0, key="cu2g",
        help="Average grade in the 2nd semester. Maximum value is 20.")
    cu2_no_eval = st.sidebar.number_input("Without Evaluations", 0, 12, 0, key="cu2n")

    st.sidebar.markdown("**👤 Demographic Information**")
    age = st.sidebar.number_input("Age at Enrollment", 17, 70, 20)

    gender = st.sidebar.selectbox("Gender", options=['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0

    marital_label = st.sidebar.selectbox(
        "Marital Status", options=list(marital_map.keys()))
    marital_status = marital_map[marital_label]

    nationality_label = st.sidebar.selectbox(
        "Nationality", options=list(nationality_map.keys()))
    nationality = nationality_map[nationality_label]

    international = st.sidebar.selectbox(
        "International Student", options=['No', 'Yes'])
    international = 1 if international == 'Yes' else 0

    displaced = st.sidebar.selectbox(
        "Displaced", options=['No', 'Yes'],
        help="Whether the student is a displaced person.")
    displaced = 1 if displaced == 'Yes' else 0

    special_needs = st.sidebar.selectbox(
        "Educational Special Needs", options=['No', 'Yes'])
    special_needs = 1 if special_needs == 'Yes' else 0

    st.sidebar.markdown("**💰 Socioeconomic Information**")
    tuition = st.sidebar.selectbox(
        "Tuition Fees Up to Date", options=['Yes', 'No'])
    tuition = 1 if tuition == 'Yes' else 0

    scholarship = st.sidebar.selectbox(
        "Scholarship Holder", options=['No', 'Yes'])
    scholarship = 1 if scholarship == 'Yes' else 0

    debtor = st.sidebar.selectbox(
        "Debtor", options=['No', 'Yes'],
        help="Whether the student has outstanding debt to the institution.")
    debtor = 1 if debtor == 'Yes' else 0

    mothers_qual_label = st.sidebar.selectbox(
        "Mother's Qualification", options=list(qualification_map.keys()))
    mothers_qual = qualification_map[mothers_qual_label]

    fathers_qual_label = st.sidebar.selectbox(
        "Father's Qualification", options=list(qualification_map.keys()))
    fathers_qual = qualification_map[fathers_qual_label]

    mothers_occ_label = st.sidebar.selectbox(
        "Mother's Occupation", options=list(occupation_map.keys()))
    mothers_occ = occupation_map[mothers_occ_label]

    fathers_occ_label = st.sidebar.selectbox(
        "Father's Occupation", options=list(occupation_map.keys()))
    fathers_occ = occupation_map[fathers_occ_label]

    st.sidebar.markdown("**🌍 Macroeconomic Indicators**")
    unemployment = st.sidebar.number_input(
        "Unemployment Rate (%)", 0.0, 25.0, 10.8,
        help="National unemployment rate at the time of enrollment.")
    inflation = st.sidebar.number_input(
        "Inflation Rate (%)", -1.0, 5.0, 1.4,
        help="National inflation rate at the time of enrollment.")
    gdp = st.sidebar.number_input(
        "GDP", -5.0, 5.0, 1.74,
        help="Gross Domestic Product at the time of enrollment.")

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

predict_clicked = st.sidebar.button("🔍 Predict Outcome")

if predict_clicked:
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
    predicted_class = le.inverse_transform(prediction)[0]

    st.markdown('<p class="section-header">📌 Prediction Result</p>',
                unsafe_allow_html=True)

    if predicted_class == "Graduate":
        st.markdown("""
            <div class="result-graduate result-card">
                <h1>✅ GRADUATE</h1>
                <p>This student is predicted to successfully complete their academic program.</p>
            </div>
        """, unsafe_allow_html=True)
    elif predicted_class == "Enrolled":
        st.markdown("""
            <div class="result-enrolled result-card">
                <h1>⚠️ ENROLLED</h1>
                <p>This student is predicted to remain enrolled but has not yet graduated.
                Academic support is recommended.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-dropout result-card">
                <h1>🚨 DROPOUT</h1>
                <p>This student is at high risk of dropping out.
                Immediate intervention is strongly recommended.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
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

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    colors = ['#FF4444', '#DAA520', '#4CAF50']
    bars = ax.bar(le.classes_, probabilities, color=colors,
                  edgecolor='white', linewidth=0.5)
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

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">📈 Model Performance & Insights</p>',
            unsafe_allow_html=True)

# ── Row 1: Model Comparison ──
st.markdown("#### 📊 Model Performance Comparison")
if os.path.exists('model_comparison.png'):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image('model_comparison.png',
                 caption='Model Performance Comparison',
                 use_container_width=True)
else:
    st.warning("Run Step 5 to generate model_comparison.png")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 2: Confusion Matrix ──
st.markdown("#### 🔲 Confusion Matrix")
if os.path.exists('confusion_matrix.png'):
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image('confusion_matrix.png',
                 caption='Confusion Matrix (Best Model)',
                 use_container_width=True)
else:
    st.warning("Run Step 5 to generate confusion_matrix.png")

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 3: Feature Importance ──
st.markdown("#### 🏆 Top 15 Feature Importances")
if os.path.exists('feature_importance.png'):
    col1, col2, col3 = st.columns([0.5, 4, 0.5])
    with col2:
        st.image('feature_importance.png',
                 caption='Top 15 Most Important Features (Best Model)',
                 use_container_width=True)
else:
    st.warning("Run Step 5 to generate feature_importance.png")

st.markdown("""
    <div class="footer">
        Student Academic Performance Prediction System &nbsp;|&nbsp;
        Amaraegbu Divine &nbsp;|&nbsp;
        Department of Computer Science &nbsp;|&nbsp; 2026
    </div>
""", unsafe_allow_html=True)
