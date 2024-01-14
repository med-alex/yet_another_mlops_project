import pandas as pd
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


tf_idf_X_train = pd.read_csv(Path.cwd()/'data'/'tf_idf_X_train.csv', converters={'text': pd.eval})
tf_idf_X_test = pd.read_csv(Path.cwd()/'data'/'tf_idf_X_test.csv', converters={'text': pd.eval})

all_tf_idf_X = pd.concat([tf_idf_X_train, tf_idf_X_test])

vec = TfidfVectorizer(min_df=2)
vec.fit([' '.join(text) for text in all_tf_idf_X.text])

tf_idf_train_emb = vec.transform([' '.join(text) for text in tf_idf_X_train.text])
tf_idf_test_emb = vec.transform([' '.join(text) for text in tf_idf_X_test.text])

with open(Path.cwd()/'data'/'tf_idf_train_embeddings.pkl', 'wb') as file:
  pickle.dump(tf_idf_train_emb, file)

with open(Path.cwd()/'data'/'tf_idf_test_embeddingspkl', 'wb') as file:
  pickle.dump(tf_idf_test_emb, file)
  