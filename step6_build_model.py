"""Creates an email classifier for identifying "interesting" e-mails."""

import cPickle
import logging
from sklearn import feature_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)

logging.info("Loading maildir data.")

with open("maildir_stemmed.pkl", "rb") as f:
    maildir_data = cPickle.load(f)

logging.info("Loading TF-IDF extractor.")

with open("maildir_tf_idf.pkl", "rb") as f:
    tf_idf_extractor = cPickle.load(f)

logging.info("Loading persons dataset.")

with open("final_project_dataset_cleaned_train.pkl", "rb") as f:
    persons_data = cPickle.load(f)

logging.info("Extracting features from %s emails.", len(maildir_data))

poi_email_addresses = set([person["email_address"] for person in persons_data
                           if person["poi"] and person["email_address"] != "NaN"])

train_features = tf_idf_extractor.transform([email["stemmed_text"] for email in maildir_data])
train_label = [email["from"] in poi_email_addresses for email in maildir_data]

logging.info("Selecting features.")

# selector = feature_selection.SelectPercentile(
#     feature_selection.f_classif, percentile=100)
# train_features = selector.fit_transform(train_features, train_label)

logging.info("Feature matrix dimensions: %s", train_features.shape)

logging.info("Found %s interesting e-mails.", len(filter(None, train_label)))

logging.info("Building e-mail classifier.")

clf = RandomForestClassifier()
clf.fit(train_features, train_label)

# pipeline = Pipeline([('selector', selector), ('rf', clf)])

logging.info("Saving e-mail classifier.")

with open("email_classifier.pkl", "wb") as f:
    cPickle.dump(clf, f)


