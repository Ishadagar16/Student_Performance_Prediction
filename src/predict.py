import joblib
import pandas as pd


MODEL_PATH = "student_performance_model.pkl"


model = joblib.load(MODEL_PATH)


def predict_performance(
    study_hours,
    attendance,
    sleep_hours,
    internet_usage,
    assignments_completed,
    previous_score
):

    student = pd.DataFrame({
        "study_hours": [study_hours],
        "attendance": [attendance],
        "sleep_hours": [sleep_hours],
        "internet_usage": [internet_usage],
        "assignments_completed": [assignments_completed],
        "previous_score": [previous_score]
    })

    prediction = model.predict(student)[0]

    if prediction == 1:
        return "Pass"
    else:
        return "Fail"



if __name__ == "__main__":

    result = predict_performance(
        study_hours=6,
        attendance=85,
        sleep_hours=7,
        internet_usage=3,
        assignments_completed=8,
        previous_score=72
    )

    print("Predicted Student Performance:", result)