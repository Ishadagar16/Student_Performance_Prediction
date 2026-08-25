import pandas as pd
import numpy as np


def load_data(file_path):
    
    df = pd.read_csv("student_dataset_10000_rows.csv")
    return df


def clean_data(df):
    
    df = df.drop_duplicates()

    df = df.dropna()

    return df


def create_target(df):
    """
    Create student performance target.

    Exam score >= 40  -> Pass
    Exam score < 40   -> Fail
    """

    df["performance"] = np.where(
        df["exam_score"] >= 40,
        "Pass",
        "Fail"
    )

    return df


def prepare_features(df):
   
    features = [
        "study_hours",
        "attendance",
        "sleep_hours",
        "internet_usage",
        "assignments_completed",
        "previous_score"
    ]

    X = df[features]

    y = df["performance"].map({
        "Fail": 0,
        "Pass": 1
    })

    return X, y


def preprocess_data(file_path):
    
    df = load_data(file_path)

    df = clean_data(df)

    df = create_target(df)

    X, y = prepare_features(df)

    return X, y, df