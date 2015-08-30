"""Extracts TF-IDF features from maildir corpus and persists it."""

import cPickle
from sklearn.feature_extraction.text import TfidfVectorizer

print "Loading maildir data."

with open("maildir_stemmed.pkl", "rb") as f:
    maildir_data = cPickle.load(f)

print "Fitting TF-IDF."

v = TfidfVectorizer(stop_words='english')
v.fit([email["stemmed_text"] for email in maildir_data])

print "Writing out extracted features."

with open("maildir_tf_idf.pkl", "wb") as f:
    cPickle.dump(v, f)
