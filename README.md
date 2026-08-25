# Student_Performance_Prediction
# Student Performance Prediction System Using Machine Learning

## 1. Project Overview

The **Student Performance Prediction System** is a Machine Learning project designed to predict the academic performance of students and identify students who may be at risk of failing.

The system analyzes factors such as study hours, attendance, sleep hours, internet usage, assignment completion, and previous academic scores. Machine Learning algorithms are then used to classify students into **Pass** or **Fail** categories.

The project is developed using Python and popular Data Science and Machine Learning libraries.

## 2. Objectives

The main objectives of this project are:

* Predict student academic performance.
* Identify students who are at risk of failing.
* Analyze factors affecting student performance.
* Help teachers identify weak students at an early stage.
* Support data-driven educational decisions.
* Compare different Machine Learning algorithms.
* Provide a reusable trained Machine Learning model.

## 3. Problem Statement

In traditional education systems, teachers may find it difficult to continuously monitor the performance of every student.

Students who are struggling academically may not always be identified early. Decisions about student performance can also depend heavily on manual observation.

This project uses historical student data and Machine Learning techniques to automatically predict whether a student is likely to pass or fail.

## 4. Proposed Solution

The proposed system performs the following tasks:

1. Loads student academic and behavioral data.
2. Cleans and preprocesses the dataset.
3. Creates a performance target based on examination scores.
4. Performs Exploratory Data Analysis.
5. Selects relevant features.
6. Splits the dataset into training and testing sets.
7. Trains multiple Machine Learning models.
8. Evaluates the models using performance metrics.
9. Selects the best-performing model.
10. Predicts the performance of new students.
11. Identifies students who may be at risk of failing.
12. Saves the trained model for future deployment.

## 5. Dataset

The project uses a dataset containing **10,000 student records**.

### Dataset Features

| Feature                 | Description                              |
| ----------------------- | ---------------------------------------- |
| `study_hours`           | Number of hours spent studying           |
| `attendance`            | Student attendance percentage            |
| `sleep_hours`           | Average daily sleep hours                |
| `internet_usage`        | Internet usage level/hours               |
| `assignments_completed` | Number or level of completed assignments |
| `previous_score`        | Previous academic score                  |
| `exam_score`            | Current examination score                |
| `placement_status`      | Student placement status                 |

### Target Variable

The target variable `performance` is created from `exam_score`.

```text
Exam Score >= 40  → Pass
Exam Score < 40   → Fail
```

The `exam_score` column is not used as an input feature because it is used to create the target variable. This prevents data leakage.

The `placement_status` column is also not used for predicting academic performance because placement is a separate outcome.

## 6. Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

### Development Tools

* Jupyter Notebook
* Visual Studio Code
* Command Prompt / Terminal
* Microsoft Excel

## 7. Machine Learning Algorithms

The project compares four Machine Learning algorithms.

### Logistic Regression

Used as a classification model to predict whether a student will pass or fail.

### Decision Tree

Uses decision rules based on student features to classify performance.

### Naive Bayes

Uses probability-based classification to predict student performance.

### Support Vector Machine

Finds a decision boundary that separates students into Pass and Fail categories.

## 8. Project Architecture

```text
Student Dataset
      |
      v
Data Collection
      |
      v
Data Preprocessing
      |
      v
Exploratory Data Analysis
      |
      v
Feature Selection
      |
      v
Train-Test Split
      |
      v
Machine Learning Models
      |
      +-------------------+
      |        |          |
      v        v          v
 Logistic   Decision    Naive Bayes
 Regression   Tree
      |
      +-------------------+
              |
              v
             SVM
              |
              v
       Model Evaluation
              |
              v
        Best Model
              |
              v
       Student Prediction
              |
       +------+------+
       |             |
       v             v
     Pass        At Risk/Fail
```

## 9. Project Structure

```text
Student_Performance_Prediction/
│
├── data/
│   └── student_dataset_10000_rows.csv
│
├── notebooks/
│   └── Student_Performance_Prediction.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── student_performance_model.pkl
│
├── outputs/
│   ├── at_risk_students.csv
│   └── student_performance_predictions.csv
│
├── requirements.txt
└── README.md
```

## 10. Project Modules

### Data Input Module

Loads the student dataset from the CSV file.

### Data Preprocessing Module

The `preprocess.py` file:

* Loads the dataset.
* Removes duplicate records.
* Handles missing values.
* Creates the Pass/Fail target.
* Selects Machine Learning features.
* Converts the target into numerical values.

### Model Training Module

The `train_model.py` file:

* Loads the processed data.
* Splits the data into training and testing sets.
* Trains multiple Machine Learning models.
* Calculates evaluation metrics.
* Compares models.
* Selects the best model.
* Saves the trained model.

### Prediction Module

The `predict.py` file:

* Loads the saved Machine Learning model.
* Accepts new student information.
* Predicts Pass or Fail.
* Can be integrated into a future web or mobile application.

## 11. Exploratory Data Analysis

The project performs EDA to understand relationships between student characteristics and academic performance.

The analysis includes:

* Exam score distribution.
* Study hours vs exam score.
* Attendance vs exam score.
* Previous score vs exam score.
* Assignment completion vs exam score.
* Sleep hours vs exam score.
* Performance distribution.
* Correlation matrix.
* Feature importance.

## 12. Model Evaluation

The Machine Learning models are evaluated using:

### Accuracy

Measures the percentage of correctly classified students.

### Precision

Measures how many students predicted as a particular class actually belong to that class.

### Recall

Measures how many actual students in a class were correctly identified.

Recall is especially important for this project because identifying students at risk of failing is one of the main objectives.

### F1 Score

The F1 Score provides a balance between precision and recall.

### Confusion Matrix

The confusion matrix shows:

```text
                 Predicted
               Fail     Pass

Actual Fail     TN       FP
Actual Pass     FN       TP
```

## 13. Installation

Make sure Python is installed on your system.

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

Alternatively, if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

## 14. Running the Project

Open Command Prompt or Terminal inside the project directory.

Move into the `src` folder:

```bash
cd src
```

Run the training program:

```bash
python train_model.py
```

This will train the Machine Learning models and save the best model in the `models` folder.

After training is complete, run:

```bash
python predict.py
```

The system will generate a prediction for the sample student provided in `predict.py`.

## 15. Running the Jupyter Notebook

Start Jupyter Notebook from the project directory:

```bash
jupyter notebook
```

Open:

```text
notebooks/Student_Performance_Prediction.ipynb
```

Run the notebook cells sequentially.

The notebook contains:

* Dataset loading
* Data preprocessing
* EDA
* Feature selection
* Model training
* Model evaluation
* Model comparison
* Student prediction
* At-risk student identification
* Model saving

## 16. Example Prediction

A new student can be represented using:

```text
Study Hours: 6
Attendance: 85
Sleep Hours: 7
Internet Usage: 3
Assignments Completed: 8
Previous Score: 72
```

The trained Machine Learning model uses these values to predict the student's performance.

Possible output:

```text
Predicted Student Performance: Pass
```

## 17. At-Risk Student Identification

Students predicted as `Fail` are considered potentially at risk.

The system can generate an at-risk student list containing relevant student information and predictions.

This information can help teachers provide:

* Additional academic support
* Extra classes
* Assignment assistance
* Individual guidance
* Performance monitoring
* Personalized learning support

## 18. Advantages

* Early identification of weak students.
* Automated performance prediction.
* Data-driven decision making.
* Helps teachers monitor student performance.
* Saves time compared with manual analysis.
* Allows comparison of multiple Machine Learning algorithms.
* Can be integrated into larger educational systems.

## 19. Limitations

* Prediction quality depends on the quality of the dataset.
* Missing or inaccurate student information can affect predictions.
* Academic performance can be influenced by factors not included in the dataset.
* Emotional, psychological, family, and social factors are not directly represented.
* A Machine Learning prediction should support educational decisions rather than replace teacher judgment.

## 20. Future Enhancements

Future versions of the project can include:

* Web-based prediction interface.
* Mobile application integration.
* Real-time student performance monitoring.
* Personalized learning recommendations.
* Automatic alerts for at-risk students.
* Integration with Learning Management Systems.
* Student dashboards.
* Teacher dashboards.
* More advanced Machine Learning and Deep Learning models.
* Historical performance tracking.

## 21. Expected Outcome

The final system provides:

* Student performance classification.
* Machine Learning model comparison.
* Performance evaluation metrics.
* Visualization of important trends.
* Identification of potentially at-risk students.
* A saved trained model that can be reused for deployment.

## 22. Conclusion

The Student Performance Prediction System demonstrates how Machine Learning can be applied to educational data to predict student academic performance.

By analyzing study hours, attendance, sleep hours, internet usage, assignment completion, and previous academic scores, the system can classify students into Pass and Fail categories.

Multiple Machine Learning algorithms are trained and evaluated to identify an effective predictive model. The resulting system can help educational institutions identify potentially weak students early and provide appropriate academic support.

The project also provides a foundation for future development into a web application, mobile application, or complete educational analytics platform.

## 23. Author

**Project:** Student Performance Prediction System Using Machine Learning

**Domain:** Data Science / Machine Learning

**Programming Language:** Python

**Dataset:** Student Academic and Behavioral Records
