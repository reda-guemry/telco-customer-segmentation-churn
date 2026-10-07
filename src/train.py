from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from src.models_pip import final_models
from src.data_loader import loader
from src.config import TARGET
from src.data_processing import get_preprocessor


from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


df = loader()
preprocessor = get_preprocessor()
models = final_models(preprocessor)




X, Y = df.drop(columns=[TARGET]), df[TARGET].map({'No' : 0, 'Yes' : 1})

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = models["logistic_regression"]

param_grid = {
    "classifier__solver": ["saga"],
    "classifier__penalty": ["elasticnet"],
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__l1_ratio": [1, 0.5, 0],
    "classifier__max_iter": [1000 , 1500, 2000 , 2500, 3000],
    "classifier__class_weight": [None, "balanced"],
}

grid_search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=50,
    scoring="recall",
    cv=5,
    random_state=42,
    n_jobs=-1,
)

model = grid_search.fit(x_train, y_train).best_estimator_


prediction = model.predict(x_test) 

result = {
    'name': "logistic_regression",
    'accuracy' : accuracy_score(y_test, prediction) , 
    'precision' : precision_score(y_test, prediction) ,
    'recall' : recall_score(y_test, prediction) , 
    'f1' : f1_score(y_test, prediction) ,
    'roc_auc' : roc_auc_score(y_test, prediction)
}

print(result)


