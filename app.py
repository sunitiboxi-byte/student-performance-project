import streamlit as st
import pandas as pd

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("🎓 Student Performance Dashboard")

df = pd.read_csv("Student_Performance_1000.csv")

# Dashboard
st.header("Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))
col2.metric("Average Final Marks", round(df["Final_Marks"].mean(),2))
col3.metric("Highest Marks", df["Final_Marks"].max())

st.write("---")

st.subheader("Student Dataset")
st.dataframe(df)

st.subheader("Top 10 Students")

top = df.sort_values(by="Final_Marks", ascending=False).head(10)

st.dataframe(top)
import plotly.express as px

st.subheader("Attendance vs Final Marks")

fig1 = px.scatter(
    df,
    x="Attendance",
    y="Final_Marks",
    color="Gender",
    title="Attendance vs Final Marks"
)

st.plotly_chart(fig1)

st.subheader("Final Marks Distribution")

fig2 = px.histogram(
    df,
    x="Final_Marks",
    nbins=20,
    title="Distribution of Final Marks"
)

st.plotly_chart(fig2)
st.subheader("Pass / Fail Analysis")

# Create Result column if it doesn't exist
df["Result"] = df["Final_Marks"].apply(lambda x: "Pass" if x >= 40 else "Fail")

result = df["Result"].value_counts()

fig3 = px.pie(
    values=result.values,
    names=result.index,
    title="Pass vs Fail"
)

st.plotly_chart(fig3)
st.subheader("⚠️ At Risk Students")

# Students with attendance below 60 or final marks below 40
risk_students = df[
    (df["Attendance"] < 60) |
    (df["Final_Marks"] < 40)
]

st.dataframe(risk_students)
st.subheader("🔍 Search Student")

student_id = st.number_input(
    "Enter Student ID",
    min_value=1,
    max_value=1000,
    step=1
)

if st.button("Search"):
    student = df[df["Student_ID"] == student_id]

    if not student.empty:
        st.write(student)
    else:
        st.error("Student Not Found")