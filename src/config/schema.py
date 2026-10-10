NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLUMNS = [
    "InternetService",
    "Contract",
    "PaymentMethod",
]

GENDER_COLUMNS = ["gender"]
TOTAL_CHARGES_COLUMN = ["TotalCharges"]

BINARY_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

DROP_COLUMNS = ["customerID", "Churn"]

TARGET = "Churn"
