# app.py
import streamlit as st
import pandas as pd
import joblib

import streamlit as st
import pandas as pd
import joblib
import cloudpickle

# -- Tambahkan semua import yang dipakai di pipeline Anda --

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# Jika Anda menggunakan imbalanced-learn di pipeline:
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline as ImbPipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)
import joblib  # for saving models/transformers

from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline as ImbPipeline

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Prediction")

# 1. Load pipeline (preprocessor + model)
#    Pastikan kedua file ini berada di direktori yang sama dengan app.py

#st.sidebar.info("Loading model…")
#with open("student_pipeline.pkl", "rb") as f:
#    pipeline = cloudpickle.load(f)
    
pipeline = joblib.load("rf_pipeline.joblib")  

# 2. Sidebar inputs untuk setiap fitur
st.sidebar.header("Input Student Features")

MARITAL_STATUS_MAP = {
    1: "Single", 2: "Married", 3: "Widower",
    4: "Divorced", 5: "Facto union", 6: "Legally separated"
}

APPLICATION_MODE_MAP = {
    1: "1st phase – general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase – special contingent (Azores)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase – special contingent (Madeira)",
    17: "2nd phase – general contingent",
    18: "3rd phase – general contingent",
    26: "Ordinance No. 533-A/99 b2 (Different Plan)",
    27: "Ordinance No. 533-A/99 b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}

COURSE_MAP = {
    33: "Biofuel Production Tech", 171: "Animation & Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy",
    9070: "Communication Design", 9085: "Veterinary Nursing",
    9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service",
    9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising & Marketing Mgmt",
    9773: "Journalism & Communication", 9853: "Basic Education",
    9991: "Management (evening)"
}

PREV_QUAL_MAP = {
    1: "Secondary education", 2: "Bachelor's degree",
    3: "Degree (1st cycle)", 4: "Master's degree",
    5: "Doctorate", 6: "Frequency of higher education",
    9: "12th year incomplete", 10: "11th year incomplete",
    12: "Other – 11th year schooling", 14: "10th year schooling",
    15: "10th year incomplete", 19: "Basic education 3rd cycle",
    38: "Basic education 2nd cycle", 39: "Technological specialization",
    40: "Higher education 1st cycle degree",
    42: "Professional higher technical", 43: "Higher education master"
}

NATIONALITY_MAP = {
    1:"Portuguese",2:"German",6:"Spanish",11:"Italian",13:"Dutch",
    14:"English",17:"Lithuanian",21:"Angolan",22:"Cape Verdean",
    24:"Guinean",25:"Mozambican",26:"Santomean",32:"Turkish",
    41:"Brazilian",62:"Romanian",100:"Moldova",101:"Mexican",
    103:"Ukrainian",105:"Russian",108:"Cuban",109:"Colombian"
}

MOTHER_QUAL_MAP = {
    1:"Secondary Education (12th Year)",2:"Bachelor's Degree",3:"Higher Education – Degree",
    4:"Master's Degree",5:"Doctorate",6:"Frequency of Higher Education",
    9:"12th Year – Not Completed",10:"11th Year – Not Completed",11:"7th Year (Old)",
    12:"Other – 11th Year",14:"10th Year",18:"General Commerce Course",
    19:"Basic Education 3rd Cycle",22:"Technical-Professional Course",
    26:"7th Year of Schooling",27:"2nd Cycle General HS",29:"9th Year – Not Completed",
    30:"8th Year",34:"Unknown",35:"Can't Read/Write",36:"Read without 4th Year",
    37:"Basic Education 1st Cycle",38:"Basic Education 2nd Cycle",
    39:"Technological Specialization Course",40:"Higher Ed 1st Cycle Degree",
    41:"Specialized Higher Studies",42:"Professional Higher Technical Course",
    43:"Higher Ed Master (2nd Cycle)",44:"Doctorate (3rd Cycle)"
}

FATHER_QUAL_MAP = {
    1:"Secondary Education (12th Year)",2:"Bachelor's Degree",3:"Higher Ed Degree",
    4:"Master's Degree",5:"Doctorate",6:"Frequency of Higher Ed",
    9:"12th Year – Not Completed",10:"11th Year – Not Completed",11:"7th Year (Old)",
    12:"Other – 11th Year",13:"2nd Year Complementary HS",14:"10th Year",
    18:"General Commerce Course",19:"Basic Ed 3rd Cycle",20:"Complementary HS Course",
    22:"Technical-Professional Course",25:"Complem. HS – Not Concluded",
    26:"7th Year of Schooling",27:"2nd Cycle General HS",29:"9th Year – Not Completed",
    30:"8th Year",31:"Admin & Commerce Course",33:"Supplementary Accounting",
    34:"Unknown",35:"Can't Read/Write",36:"Read without 4th Year",
    37:"Basic Ed 1st Cycle",38:"Basic Ed 2nd Cycle",39:"Tech Spec Course",
    40:"Higher Ed 1st Cycle",41:"Spec. Higher Studies",42:"Professional Higher Technical",
    43:"Higher Ed Master (2nd Cycle)",44:"Doctorate (3rd Cycle)"
}

MOTHER_OCCUP_MAP = {
     0:"Student",1:"Legislative/Exec Directors",2:"Intellectual/Scientific Specialists",
     3:"Technicians/Professions",4:"Administrative Staff",5:"Personal/Service/Sales",
     6:"Farmers/Agri/Fish/Forest Workers",7:"Industry/Construction Workers",
     8:"Machine Operators/Assemblers",9:"Unskilled Workers",10:"Armed Forces",
     90:"Other",99:"Blank",122:"Health Professionals",123:"Teachers",
    125:"ICT Specialists",131:"Science/Engineering Technicians",
    132:"Health Technicians",134:"Legal/Social/Cultural Technicians",
    141:"Office Workers/Data Operators",143:"Financial/Registry Operators",
    144:"Admin Support Staff",151:"Personal Service Workers",152:"Sellers",
    153:"Personal Care Workers",171:"Construction Workers",173:"Printing/Precision",
    175:"Food/Wood/Clothing Industr. Workers",191:"Cleaning Workers",
    192:"Unskilled Agri/Fish/Forest",193:"Unskilled Extractive/Transport",
    194:"Meal Preparation Assistants"
}

FATHER_OCCUP_MAP = {
     0:"Student",1:"Legislative/Exec Directors",2:"Intellectual/Scientific Specialists",
     3:"Technicians/Professions",4:"Administrative Staff",5:"Personal/Service/Sales",
     6:"Farmers/Agri/Fish/Forest Workers",7:"Industry/Construction Workers",
     8:"Machine Operators/Assemblers",9:"Unskilled Workers",10:"Armed Forces",
     90:"Other",99:"Blank",101:"Armed Forces Officers",102:"Armed Forces Sergeants",
    103:"Other Armed Forces",112:"Admin/Commercial Directors",114:"Hotel/Catering Directors",
    121:"Physical Sciences/Engineering Specialists",122:"Health Professionals",
    123:"Teachers",124:"Finance/Accounting Specialists",131:"Science/Eng Techs",
    132:"Health Techs",134:"Legal/Social/Cultural Techs",135:"ICT Techs",
    141:"Office/Data Operators",143:"Finance/Registry Ops",144:"Admin Support",
    151:"Personal Service Workers",152:"Sellers",153:"Personal Care Workers",
    154:"Protection/Security Services",161:"Market-Oriented Farmers",163:"Subsistence Farmers",
    171:"Skilled Construction",172:"Metalworking",174:"Electric/Electronics",
    175:"Food/Wood/Clothing Workers",181:"Plant/Machine Ops",182:"Assembly Workers",
    183:"Vehicle Drivers",192:"Unskilled Agri/Fish/Forest",193:"Unskilled Extractive",
    194:"Meal Prep Assistants",195:"Street Vendors"
}



marital_status = st.sidebar.selectbox(
    "Marital Status", options=list(MARITAL_STATUS_MAP.keys()),
    format_func=lambda x: MARITAL_STATUS_MAP[x]
)

application_mode = st.sidebar.selectbox(
    "Application Mode", options=list(APPLICATION_MODE_MAP.keys()),
    format_func=lambda x: APPLICATION_MODE_MAP[x]
)

application_order = st.sidebar.slider(
    "Application Order", 0, 9, 0
)

course = st.sidebar.selectbox(
    "Course", options=list(COURSE_MAP.keys()),
    format_func=lambda x: COURSE_MAP[x]
)

daytime_evening = st.sidebar.selectbox(
    "Day vs Evening", options=[1,0],
    format_func=lambda x: "Daytime" if x==1 else "Evening"
)


previous_qualification = st.sidebar.selectbox(
    "Previous Qualification",
    options=list(PREV_QUAL_MAP.keys()),
    format_func=lambda x: PREV_QUAL_MAP[x]
)

previous_grade = st.sidebar.slider(
    "Previous Qualification Grade", 0.0, 200.0, 120.0
)

nationality = st.sidebar.selectbox(
    "Nationality", options=list(NATIONALITY_MAP.keys()),
    format_func=lambda x: NATIONALITY_MAP[x]
)


mother_qualification = st.sidebar.selectbox(
    "Mother's Qualification", options=list(MOTHER_QUAL_MAP.keys()),
    format_func=lambda x: MOTHER_QUAL_MAP[x]
)

father_qualification = st.sidebar.selectbox(
    "Father's Qualification", options=list(FATHER_QUAL_MAP.keys()),
    format_func=lambda x: FATHER_QUAL_MAP[x]
)

mother_occupation = st.sidebar.selectbox(
    "Mother's Occupation", options=list(MOTHER_OCCUP_MAP.keys()),
    format_func=lambda x: MOTHER_OCCUP_MAP[x]
)

father_occupation = st.sidebar.selectbox(
    "Father's Occupation", options=list(FATHER_OCCUP_MAP.keys()),
    format_func=lambda x: FATHER_OCCUP_MAP[x]
)

admission_grade = st.sidebar.slider("Admission grade", 0.0, 200.0, 130.0)

displaced = st.sidebar.selectbox("Displaced", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
educational_needs = st.sidebar.selectbox("Educational special needs", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
debtor = st.sidebar.selectbox("Debtor", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
tuition_up_to_date = st.sidebar.selectbox("Tuition fees up to date", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
gender = st.sidebar.selectbox("Gender", [0,1], format_func=lambda x: "Female" if x==0 else "Male")
scholarship_holder = st.sidebar.selectbox("Scholarship holder", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

age_at_enrollment = st.sidebar.number_input("Age at enrollment", min_value=15, max_value=100, value=20)

international = st.sidebar.selectbox("International student", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

curr1_credited   = st.sidebar.number_input("1st sem credited",   min_value=0, value=0)
curr1_enrolled   = st.sidebar.number_input("1st sem enrolled",   min_value=0, value=0)
curr1_evaluated  = st.sidebar.number_input("1st sem evaluations",min_value=0, value=0)
curr1_approved   = st.sidebar.number_input("1st sem approved",   min_value=0, value=0)

unemployment_rate = st.sidebar.number_input(
    "Unemployment rate (%)", min_value=0.0, value=10.0
)
inflation_rate = st.sidebar.number_input(
    "Inflation rate (%)", min_value=-10.0, value=1.4
)
gdp = st.sidebar.number_input(
    "GDP growth (%)", min_value=-10.0, value=1.74
)

# 1st sem extra
curr1_grade    = st.sidebar.number_input(
    "1st sem average grade", min_value=0.0, max_value=20.0, value=14.0
)
curr1_noeval   = st.sidebar.number_input(
    "1st sem without evaluations", min_value=0, value=0
)

# 2nd sem all
curr2_credited   = st.sidebar.number_input(
    "2nd sem credited", min_value=0, value=0
)
curr2_enrolled   = st.sidebar.number_input(
    "2nd sem enrolled", min_value=0, value=0
)
curr2_evaluated  = st.sidebar.number_input(
    "2nd sem evaluations", min_value=0, value=0
)
curr2_approved   = st.sidebar.number_input(
    "2nd sem approved", min_value=0, value=0
)
curr2_grade      = st.sidebar.number_input(
    "2nd sem average grade", min_value=0.0, max_value=20.0, value=13.0
)
curr2_noeval     = st.sidebar.number_input(
    "2nd sem without evaluations", min_value=0, value=0
)

# 3. Buat DataFrame input
input_dict = {
    'Marital_status': marital_status,
    'Application_mode': application_mode,
    'Application_order': application_order,
    'Course': course,
    'Daytime_evening_attendance': daytime_evening,
    'Previous_qualification': previous_qualification,
    'Previous_qualification_grade': previous_grade,
    'Nacionality': nationality,
    "Mothers_qualification": mother_qualification,
    "Fathers_qualification": father_qualification,
    "Mothers_occupation": mother_occupation,
    "Fathers_occupation": father_occupation,
    "Admission_grade": admission_grade,
    "Displaced": displaced,
    "Educational_special_needs": educational_needs,
    "Debtor": debtor,
    "Tuition_fees_up_to_date": tuition_up_to_date,
    "Gender": gender,
    "Scholarship_holder": scholarship_holder,
    "Age_at_enrollment": age_at_enrollment,
    "International": international,
    "Curricular_units_1st_sem_credited": curr1_credited,
    "Curricular_units_1st_sem_enrolled": curr1_enrolled,
    "Curricular_units_1st_sem_evaluations": curr1_evaluated,
    "Curricular_units_1st_sem_approved": curr1_approved,
    'Unemployment_rate': unemployment_rate,
    'Inflation_rate': inflation_rate,
    'GDP': gdp,
    'Curricular_units_1st_sem_grade': curr1_grade,
    'Curricular_units_1st_sem_without_evaluations': curr1_noeval,
    'Curricular_units_2nd_sem_credited': curr2_credited,
    'Curricular_units_2nd_sem_enrolled': curr2_enrolled,
    'Curricular_units_2nd_sem_evaluations': curr2_evaluated,
    'Curricular_units_2nd_sem_approved': curr2_approved,
    'Curricular_units_2nd_sem_grade': curr2_grade,
    'Curricular_units_2nd_sem_without_evaluations': curr2_noeval
}

input_df = pd.DataFrame([input_dict])

# 4. Button predict
if st.button("Predict"):
    pred   = pipeline.predict(input_df)[0]
    proba  = pipeline.predict_proba(input_df)[0]
    labels = ['Graduate','Dropout','Enrolled']
    st.subheader("Prediction Result")
    st.write(f"🏷️ **Status:** {labels[pred]}")
    st.write(f"📊 **Probabilities:**")
    st.write(f"- Graduate: {proba[0]:.2f}")
    st.write(f"- Dropout : {proba[1]:.2f}")
    st.write(f"- Enrolled: {proba[2]:.2f}")
