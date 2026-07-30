import pandas as pd

# Read the CSV file
df = pd.read_csv("Student_Performance_1000.csv")

# Display the first 5 rows
print("===== STUDENT PERFORMANCE DATA =====")
print(df.head())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display statistical summary
print("\nStatistical Summary:")
print(df.describe())
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Save cleaned dataset
df.to_csv("Student_Performance_Cleaned.csv", index=False)

print("\nData Cleaning Completed!")
import matplotlib.pyplot as plt

# 1. Bar Chart - Student Final Marks
plt.figure(figsize=(10,5))
plt.bar(df["Student_ID"], df["Final_Marks"])
plt.title("Student Final Marks")
plt.xlabel("Student ID")
plt.ylabel("Final Marks")
plt.show()

# 2. Histogram - Final Marks
plt.figure(figsize=(8,5))
plt.hist(df["Final_Marks"], bins=10)
plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Number of Students")
plt.show()

# 3. Scatter Plot - Attendance vs Final Marks
plt.figure(figsize=(8,5))
plt.scatter(df["Attendance"], df["Final_Marks"])
plt.title("Attendance vs Final Marks")
plt.xlabel("Attendance")
plt.ylabel("Final Marks")
plt.show()

# 4. Box Plot - Final Marks
plt.figure(figsize=(6,5))
plt.boxplot(df["Final_Marks"])
plt.title("Box Plot of Final Marks")
plt.ylabel("Final Marks")
plt.show()
# -----------------------------
# Feature Engineering
# -----------------------------

# Create Average Marks
df["Average_Marks"] = (
    df["Assignment_Marks"] +
    df["Midterm_Marks"] +
    df["Final_Marks"]
) / 3

# Create Performance Category
def performance(mark):
    if mark >= 80:
        return "Excellent"
    elif mark >= 60:
        return "Good"
    elif mark >= 40:
        return "Average"
    else:
        return "Poor"

df["Performance"] = df["Average_Marks"].apply(performance)

print("\nFeature Engineering Completed!")
print(df[["Average_Marks", "Performance"]].head())
# -----------------------------
# Machine Learning Model
# -----------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Input Features
X = df[["Attendance", "Assignment_Marks", "Study_Hours_Per_Day", "Midterm_Marks"]]

# Output
y = df["Final_Marks"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

print("\nMachine Learning Model Trained Successfully!")

# Predict
predictions = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(predictions[:5])
# -----------------------------
# Model Evaluation
# -----------------------------

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Evaluate the model
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("\nModel Evaluation")
print("R2 Score:", round(r2, 4))
print("Mean Absolute Error:", round(mae, 2))
print("Mean Squared Error:", round(mse, 2))
# -----------------------------
# Student Performance Prediction
# -----------------------------

attendance = 90
assignment = 85
study_hours = 6
midterm = 80

new_student = [[attendance, assignment, study_hours, midterm]]

predicted_marks = model.predict(new_student)

print("\nStudent Performance Prediction")
print("Attendance:", attendance)
print("Assignment Marks:", assignment)
print("Study Hours Per Day:", study_hours)
print("Midterm Marks:", midterm)
print("Predicted Final Marks:", round(predicted_marks[0], 2))
# -----------------------------
# Risk Detection Module
# -----------------------------

def check_risk(attendance, predicted_marks):
    if attendance < 60 or predicted_marks < 40:
        return "At Risk"
    else:
        return "Safe"

risk = check_risk(attendance, predicted_marks[0])

print("\nRisk Detection")
print("Risk Status:", risk)
# -----------------------------
# Top 10 Students
# -----------------------------

print("\nTop 10 Students")

top_students = df.sort_values(
    by="Final_Marks",
    ascending=False
).head(10)

print(top_students[[
    "Student_ID",
    "Name",
    "Attendance",
    "Final_Marks"
]])
# -----------------------------
# Pass / Fail Analysis
# -----------------------------

# Create Result column
df["Result"] = df["Final_Marks"].apply(
    lambda x: "Pass" if x >= 40 else "Fail"
)

print("\nPass / Fail Summary")

print(df["Result"].value_counts())

# Pie Chart
import matplotlib.pyplot as plt

result = df["Result"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    result,
    labels=result.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Pass vs Fail")
plt.show()