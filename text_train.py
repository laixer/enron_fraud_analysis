import cPickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

print "Loading emails."

with open('non_test_emails.pkl', 'rb') as f:
    non_test_emails = cPickle.load(f)

is_poi_data = []
word_data = []

for email in non_test_emails:
    word_data.append(email["stemmed_words"])
    is_poi_data.append(email["all_poi"])

print "Vectorizing data."

v = TfidfVectorizer(stop_words='english', max_df=0.5)
features_train = v.fit_transform(word_data)

selector = SelectPercentile(f_classif, percentile=1)
selector.fit(features_train_transformed, labels_train)
features_train_transformed = selector.transform(features_train_transformed).toarray()
features_test_transformed  = selector.transform(features_test_transformed).toarray()

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, is_poi_data)
