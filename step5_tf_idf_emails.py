"""Extracts TF-IDF features from maildir corpus and persists it."""

import cPickle
import logging
from sklearn import feature_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)

logging.info("Loading maildir data.")

with open("maildir_stemmed.pkl", "rb") as f:
    maildir_data = cPickle.load(f)

logging.info("Fitting TF-IDF.")

v = TfidfVectorizer(stop_words='english', max_df=0.5, sublinear_tf=True)
v.fit([email["stemmed_text"] for email in maildir_data])

logging.info("Writing out feature extractor.")

with open("maildir_tf_idf.pkl", "wb") as f:
    cPickle.dump(v, f)
