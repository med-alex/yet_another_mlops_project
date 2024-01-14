from sklearn.metrics import accuracy_score, f1_score
import pickle
import pandas as pd
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
import os


os.environ["MLFLOW_REGISTRY_URI"] = str(Path.cwd()/'mlflow')
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_transf_model")

with open(Path.cwd()/'data'/'transf_test_embeddings.pkl', 'rb') as file:
  test_embeddings = pickle.load(file)
Y_test = pd.read_csv(Path.cwd()/'data'/'transf_Y_test.csv')

client = MlflowClient()
exp = client.get_experiment_by_name('get_best_transf_xgb_params')
runs_info = client.search_runs(exp.experiment_id)

clf_model = mlflow.xgboost.load_model(f"runs:/{runs_info[0].info.run_id}/best_transf_xgboost")
pred = clf_model.predict(test_embeddings)

accuracy = round(accuracy_score(pred, Y_test), 3)
f1 = round(f1_score(pred, Y_test), 3)

with mlflow.start_run():
    mlflow.log_input(mlflow.data.from_numpy(test_embeddings, 
                                            targets=Y_test.iloc[:, 0].values), 
                     context="test")
    mlflow.log_param('model_params', clf_model.get_params())
    mlflow.log_metric('test_accuracy_score', accuracy)
    mlflow.log_metric('test_f1_score', f1)
