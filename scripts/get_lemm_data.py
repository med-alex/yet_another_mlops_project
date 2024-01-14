import pandas as pd
from pathlib import Path
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from functions.data_preprocessing import lemmatization, remove_most_common_words


prep_data = pd.read_csv(Path.cwd()/'data'/'prep_data.csv', converters={'text': pd.eval})

en_lemmatizer = WordNetLemmatizer()

en_stopwords = stopwords.words("english")
additional_stop_words = ['\\a', '\x89û', 'å', 'å£', 'å¤', 'å¨', 'åç', 'åè', 'åê', 'ì¢', 'ìñ', 'ìü']
for word in additional_stop_words:
  en_stopwords.append(word)

lemm_data = prep_data.text.apply(lambda text: lemmatization(text,
                                                            en_stopwords,
                                                            en_lemmatizer))

clean_lemm_data = remove_most_common_words(lemm_data, 30)

clean_lemm_data.to_csv(Path.cwd()/'data'/'lemm_data.csv', index=False)
