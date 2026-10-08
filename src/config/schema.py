

NUMERICAL_COLUMNS = ['tenure', 'MonthlyCharges', 'TotalCharges' ]

CATEGORICAL_COLUMNS = ['MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies','Contract', 'PaymentMethod']

GENDER_COLUMNS = ['gender']
TOTAL_CHARGES_COLUMN = ['TotalCharges']

BINARY_COLUMNS = ['SeniorCitizen' , 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling' ]

DROP_COLUMNS = ['customerID', 'Churn']

TARGET = 'Churn'
