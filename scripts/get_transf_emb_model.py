from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle

emb_model = SentenceTransformer('all-mpnet-base-v1')
emb_model.max_seq_length = 64

with open(Path.cwd()/'models'/'emb_model.pkl', 'wb') as file:
  pickle.dump(emb_model, file)
