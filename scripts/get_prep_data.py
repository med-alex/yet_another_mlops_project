from functions.data_preprocessing import preprocessing
import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


use_data = pd.read_csv(Path.cwd()/'data'/'use_data.csv', converters={'text': pd.eval})

en_stopwords = stopwords.words("english")
additional_stop_words = ['\\a', '\x89û', 'å', 'å£', 'å¤', 'å¨', 'åç', 'åè', 'åê', 'ì¢', 'ìñ', 'ìü']
for word in additional_stop_words:
  en_stopwords.append(word)

prep_data = use_data.text.apply(lambda text: preprocessing(text, en_stopwords))
prep_data = prep_data.drop_duplicates()
labels = pd.Series([prep_data.target.loc[index] for index in prep_data.index], index=prep_data.index)

prep_data.to_csv(Path.cwd()/'data'/'prep_data.csv', index=False)
labels.to_csv(Path.cwd()/'data'/'labels.csv', index=False)
