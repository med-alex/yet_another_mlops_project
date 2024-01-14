import pickle
import pandas as pd
from pathlib import Path
import mlflow
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import os


os.environ["MLFLOW_REGISTRY_URI"] = str(Path.cwd()/'mlflow')
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("get_best_xgb_params")

with open(Path.cwd()/'data'/'train_embeddings.pkl', 'rb') as file:
  train_embeddings = pickle.load(file)
Y_train = pd.read_csv(Path.cwd()/'data'/'Y_train.csv')

param_grid = {'n_estimators': [200, 300, 400],
              'max_depth': [6, 8, 10]}

clf_model = XGBClassifier()

opt = GridSearchCV(clf_model,
                   param_grid, scoring=['f1', 'accuracy'], cv=2,
                   error_score='raise', refit='accuracy')
opt.fit(train_embeddings, Y_train)
best_model = opt.best_estimator_

with mlflow.start_run():
    mlflow.log_input(mlflow.data.from_numpy(train_embeddings, 
                                            targets=Y_train.iloc[:, 0].values), 
                     context="training")
    mlflow.log_param('model_type', clf_model.__class__)
    mlflow.log_param('param_grid', param_grid)
    mlflow.log_param('best_params', opt.best_params_)
    mlflow.log_metric('best_train_accuracy_score', round(opt.best_score_, 3))
    mlflow.xgboost.log_model(best_model,
                             artifact_path='best_xgboost',
                             registered_model_name='best_xgboost')
    mlflow.log_artifact(local_path=Path.cwd()/'scripts'/'get_best_xgb_params.py',
                        artifact_path='get_best_xgb_params code')
