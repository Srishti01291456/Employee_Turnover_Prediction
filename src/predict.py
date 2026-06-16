import pickle
import pandas as pd

model = pickle.load(
    open("../models/turnover_model.pkl", "rb")
)

scaler = pickle.load(
    open("../models/scaler.pkl", "rb")
)

sample_employee = pd.DataFrame({
    "Job_Satisfaction":[4],
    "Performance_Rating":[4],
    "Years_At_Company":[5],
    "Work_Life_Balance":[4],
    "Distance_From_Home":[10],
    "Monthly_Income":[60000],
    "Education_Level":[3],
    "Age":[30],
    "Companies_Worked":[2],
    "Employee_Role":[5],
    "Annual_Bonus":[100000],
    "Training_Hours":[20],
    "Department":[5]
})

sample_scaled = scaler.transform(
    sample_employee
)

prediction = model.predict(
    sample_scaled
)

if prediction[0] == 1:
    print("Employee likely to leave")
else:
    print("Employee likely to stay")
