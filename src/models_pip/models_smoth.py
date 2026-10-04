

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import  XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def models_smote_pip(preprocessor) :  
    models = {
        'logistic_regression' : Pipeline([
            ('preprocessor', preprocessor), 
            ('smote' , SMOTE(random_state=42)),
            ('classifier', LogisticRegression())
        ]),
        'decision_tree' : Pipeline([
            ('preprocessor', preprocessor),
            ('smote' , SMOTE(random_state=42)),
            ('classifier', DecisionTreeClassifier())
        ]),
        'random_forest' : Pipeline([
            ('preprocessor', preprocessor),
            ('smote' , SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier())
        ]),
        'SVR' : Pipeline([
            ('preprocessor', preprocessor),
            ('smote' , SMOTE(random_state=42)),
            ('classifier', SVC(probability=True))
        ]),
        'XGBoost' : Pipeline([
            ('preprocessor', preprocessor),
            ('smote' , SMOTE(random_state=42)), 
            ('classifier', XGBClassifier())
        ])
    }

    return models

def models_pip(preprocessor) :

    models = {
        'logistic_regression' : Pipeline([
            ('preprocessor', preprocessor), 
            ('classifier', LogisticRegression())
        ]),
        'decision_tree' : Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier())
        ]),
        'random_forest' : Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier())
        ]),
        'SVR' : Pipeline([
            ('preprocessor', preprocessor),
            ('smote' , SMOTE(random_state=42)),
            ('classifier', SVC(probability=True))
        ]),
        'XGBoost' : Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier())
        ])
    }

    return models 