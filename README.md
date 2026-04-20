# Model Deployment - Student Placement & Salary Prediction

This project is a monolithic machine learning deployment using Streamlit. It predicts student placement status and estimated salary based on academic and skill-related features.

## Project Description

This application allows users to input student data and obtain predictions for:

- Placement Status (Classification)
- Expected Salary (Regression)

The system is implemented in a single Streamlit application (monolithic architecture).

## Dataset Features

The model uses the following input features:

- Gender
- Branch
- CGPA
- 10th Percentage
- 12th Percentage
- Backlogs
- Study Hours per Day
- Attendance Percentage
- Projects Completed
- Internships Completed
- Coding Skill Rating
- Communication Skill Rating
- Aptitude Skill Rating
- Hackathons Participated
- Certifications Count
- Sleep Hours
- Stress Level
- Part Time Job
- Family Income Level
- City Tier
- Internet Access
- Extracurricular Involvement

## Models Used

- Classification Model: clf.pkl
- Regression Model: reg.pkl

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib

## How to Run Locally

streamlit run app_streamlit.py

## Deployment

This application is deployed using Streamlit Cloud.

Live Application:
(Insert Streamlit URL here after deployment)

## Project Structure

model-deployment-StudentPrediction/
├── app_streamlit.py
├── clf.pkl
├── reg.pkl
├── requirements.txt
├── README.md

## Author

Student Project - Model Deployment Assignment