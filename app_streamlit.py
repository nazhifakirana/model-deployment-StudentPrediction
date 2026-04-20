import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_models():
    clf = joblib.load("clf.pkl")
    reg = joblib.load("reg.pkl")
    return clf, reg

clf, reg = load_models()

st.set_page_config(page_title="Student Prediction System", layout="wide")

st.title("🎓 Student Placement & Salary Prediction System")

st.sidebar.title("📌 Input Guide")
st.sidebar.info(
    "Aplikasi ini digunakan untuk memprediksi:\n"
    "- 🎯 Status Placement (Lulus / Tidak)\n"
    "- 💰 Estimasi Salary (LPA)\n\n"
    "Silakan isi semua data pada form utama."
)

st.sidebar.markdown("---")
st.sidebar.success("🤖 Model Ready (Random Forest + Gradient Boosting)")
st.sidebar.caption("Pastikan input sesuai data mahasiswa")

st.header("🧾 Input Student Data")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📚 Academic")
    gender = st.selectbox("Gender", ["Male", "Female"])
    branch = st.selectbox("Branch", ["CS", "IT", "ECE", "EEE", "MECH"])
    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.0)
    tenth_percentage = st.number_input("10th Percentage", 0.0, 100.0, 70.0)
    twelfth_percentage = st.number_input("12th Percentage", 0.0, 100.0, 70.0)
    backlogs = st.number_input("Backlogs", 0, 10, 0)

with col2:
    st.subheader("🧠 Skills")
    coding_skill_rating = st.number_input("Coding Skill", 0, 10, 5)
    communication_skill_rating = st.number_input("Communication Skill", 0, 10, 5)
    aptitude_skill_rating = st.number_input("Aptitude Skill", 0, 10, 5)

    projects_completed = st.number_input("Projects", 0, 20, 2)
    internships_completed = st.number_input("Internships", 0, 10, 1)
    hackathons_participated = st.number_input("Hackathons", 0, 10, 0)
    certifications_count = st.number_input("Certifications", 0, 10, 1)

with col3:
    st.subheader("🧘 Lifestyle")
    study_hours_per_day = st.number_input("Study Hours", 0.0, 24.0, 5.0)
    attendance_percentage = st.number_input("Attendance %", 0.0, 100.0, 75.0)
    sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
    stress_level = st.number_input("Stress Level", 0, 10, 5)

    part_time_job = st.selectbox("Part Time Job", ["Yes", "No"])
    family_income_level = st.selectbox("Family Income", ["Low", "Medium", "High"])
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet_access = st.selectbox("Internet Access", ["Yes", "No"])
    extracurricular_involvement = st.selectbox("Extracurricular", ["Low", "Medium", "High"])

st.markdown("---")

if st.button("🚀 Predict"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "branch": branch,
        "cgpa": cgpa,
        "tenth_percentage": tenth_percentage,
        "twelfth_percentage": twelfth_percentage,
        "backlogs": backlogs,
        "study_hours_per_day": study_hours_per_day,
        "attendance_percentage": attendance_percentage,
        "projects_completed": projects_completed,
        "internships_completed": internships_completed,
        "coding_skill_rating": coding_skill_rating,
        "communication_skill_rating": communication_skill_rating,
        "aptitude_skill_rating": aptitude_skill_rating,
        "hackathons_participated": hackathons_participated,
        "certifications_count": certifications_count,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "part_time_job": part_time_job,
        "family_income_level": family_income_level,
        "city_tier": city_tier,
        "internet_access": internet_access,
        "extracurricular_involvement": extracurricular_involvement
    }])

    placement = clf.predict(input_data)[0]
    salary = reg.predict(input_data)[0]

    st.subheader("📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        if placement == 1:
            st.success("🎉 PLACED")
        else:
            st.error("❌ NOT PLACED")

    with colB:
        st.info(f"💰 Salary Prediction: {round(salary, 2)} LPA")

    st.subheader("📈 Visualization")

    chart_df = pd.DataFrame({
        "Feature": [
            "CGPA", "10th %", "12th %", "Study Hours",
            "Attendance", "Projects", "Internships",
            "Coding", "Communication", "Aptitude", "Stress"
        ],
        "Value": [
            cgpa, tenth_percentage, twelfth_percentage,
            study_hours_per_day, attendance_percentage,
            projects_completed, internships_completed,
            coding_skill_rating, communication_skill_rating,
            aptitude_skill_rating, stress_level
        ]
    })

    st.bar_chart(chart_df.set_index("Feature"))

    st.subheader("🧾 Input Summary")
    st.dataframe(input_data)
