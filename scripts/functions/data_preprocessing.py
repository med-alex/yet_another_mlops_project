import contractions
import re
import emoji
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from collections import Counter


def preprocessing(text, stopwords):

  text = ' '.join([contractions.fix(word) for word in text.split()])
  text = ' ' + text + ' '
  text = ' '.join([word for word in text.split() if emoji.is_emoji(word) == False])
  text = ' ' + text + ' '
  text = re.sub(r"[{}]".format(string.punctuation) + '|\n|\t', ' ', text)
  while re.findall(r'\shttp\S*\s|\swww\S*\s', text):
    text = re.sub(r'\shttp\S*\s|\swww\S*\s', ' ', text)
  while re.findall(r'\sRe\s|\sre\s', text):
    text = re.sub(r'\sRe\s|\sre\s', ' ', text)
  text = re.sub('\d', ' ', text)
  text = re.sub('\s+', ' ', text)
  text = text.lower()
  text = text.strip()
  text = ' '.join([word for word in text.split() if word not in stopwords])

  return text


def lemmatization(text, stopwords, lemm_model):

  text = word_tokenize(text)
  text = [lemm_model.lemmatize(word) for word in text]
  text = [word for word in text if word not in stopwords]

  return text


def remove_most_common_words(data, num):

  overall_text = []
  for text in data:
    overall_text += text

  most_common_words = Counter((' '.join(overall_text)).split())\
                                                    .most_common(num)

  clean_data = data.apply(lambda text: \
                    [word for word in text if word not in [word for word, count in most_common_words]])

  return clean_data
