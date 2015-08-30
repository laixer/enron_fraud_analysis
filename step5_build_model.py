"""Creates an email classifier for identifying "interesting" e-mails."""

import cPickle
from sklearn.ensemble import RandomForestClassifier

print "Loading maildir data."

with open("maildir_stemmed.pkl", "rb") as f:
    maildir_data = cPickle.load(f)

print "Loading TF-IDF extractor."

with open("maildir_tf_idf.pkl", "rb") as f:
    tf_idf_extractor = cPickle.load(f)

print "Loading persons dataset."

with open("final_project_dataset_cleaned_train.pkl", "rb") as f:
    persons_data = cPickle.load(f)

print "Extracting features."

poi_email_addresses = set([person["email_address"] for person in persons_data
                           if person["poi"] and person["email_address"] != "NaN"])

train_features = tf_idf_extractor.transform([email["stemmed_text"] for email in maildir_data])
train_label = [email["recipients"].issubset(poi_email_addresses) for email in maildir_data]

print "Building e-mail classifier."

clf = RandomForestClassifier()
clf.fit(train_features, train_label)

print "Saving e-mail classifier."

with open("email_classifier.pkl", "wb") as f:
    cPickle.dump(clf, f)

