import pandas as pd
from pathlib import Path
import pickle


transf_X_train = pd.read_csv(Path.cwd()/'data'/'transf_X_train.csv', converters={'text': pd.eval})
transf_X_test = pd.read_csv(Path.cwd()/'data'/'transf_X_test.csv', converters={'text': pd.eval})

with open(Path.cwd()/'models'/'transf_emb_model.pkl', 'rb') as file:
  emb_model = pickle.load(file)

train_embeddings = emb_model.encode(transf_X_train.text)
test_embeddings = emb_model.encode(transf_X_test.text)

with open(Path.cwd()/'data'/'transf_train_embeddings.pkl', 'wb') as file:
  pickle.dump(train_embeddings, file)

with open(Path.cwd()/'data'/'transf_test_embeddings.pkl', 'wb') as file:
  pickle.dump(test_embeddings, file)
