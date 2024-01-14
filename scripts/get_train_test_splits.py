import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


prep_data = pd.read_csv(Path.cwd()/'data'/'prep_data.csv', converters={'text': pd.eval})
labels = pd.read_csv(Path.cwd()/'data'/'labels.csv', converters={'text': pd.eval})

transf_X_train, transf_X_test, transf_Y_train, transf_Y_test = train_test_split(prep_data, labels, test_size=0.2)

transf_X_train.to_csv(Path.cwd()/'data'/'transf_X_train.csv', index=False)
transf_X_test.to_csv(Path.cwd()/'data'/'transf_X_test.csv', index=False)
transf_Y_train.to_csv(Path.cwd()/'data'/'transf_Y_train.csv', index=False)
transf_Y_test.to_csv(Path.cwd()/'data'/'transf_Y_test.csv', index=False)

tf_idf_data = pd.read_csv(Path.cwd()/'data'/'tf_idf_data.csv', converters={'text': pd.eval})

tf_idf_X_train, tf_idf_X_test, tf_idf_Y_train, tf_idf_Y_test = train_test_split(tf_idf_data, labels, test_size=0.2)

tf_idf_X_train.to_csv(Path.cwd()/'data'/'tf_idf_X_train.csv', index=False)
tf_idf_X_test.to_csv(Path.cwd()/'data'/'tf_idf_X_test.csv', index=False)
tf_idf_Y_train.to_csv(Path.cwd()/'data'/'tf_idf_Y_train.csv', index=False)
tf_idf_Y_test.to_csv(Path.cwd()/'data'/'tf_idf_Y_test.csv', index=False)
