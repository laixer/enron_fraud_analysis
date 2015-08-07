#!/usr/bin/python
import email
import os
from pprint import pprint
import string

import sys
import pickle
import cPickle
from sklearn import grid_search
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import SnowballStemmer

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from parse_out_email_text import parseOutText

from tester import test_classifier, dump_classifier_and_data


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

def write_data_dict_to_csv(file):
    all_features = data_dict["BAXTER JOHN C"].keys()
    with open(file, "w") as f:
        f.write("name," + string.join(all_features, ",") + "\n")
        for person, person_data in data_dict.iteritems():
            person_data_values = list(map(lambda feature: str(person_data[feature]), all_features))
            f.write(person + "," + string.join(person_data_values, ",") + "\n")

### Task 2: Remove outliers
del data_dict['TOTAL']  # Summary row from financial data
del data_dict['BELFER ROBERT']  # Data incorrectly imported

write_data_dict_to_csv("final_project_dataset.csv")

### Task 3: Create new feature(s)
for k, v in data_dict.iteritems():
    from_this_person_to_poi = data_dict[k]["from_this_person_to_poi"]
    from_messages = data_dict[k]["from_messages"]

    if from_this_person_to_poi == 'NaN' or from_messages == 'NaN':
        from_this_person_to_poi_pct = 'NaN'
    else:
        from_this_person_to_poi_pct = str(int(round((float(from_this_person_to_poi) / float(from_messages)) * 10)))

    # print from_this_person_to_poi
    # print from_messages
    # print from_this_person_to_poi_pct
    data_dict[k]["from_this_person_to_poi_pct"] = from_this_person_to_poi_pct

features_list = [
    'poi',
    # 'deferral_payments',
    'exercised_stock_options',
    'bonus',
    'total_stock_value',
    'deferred_income'
]

matches = []

poi_email_addresses = set()
for name, data in data_dict.iteritems():
    if data['poi']:
        poi_email_addresses.add(data['email_address'])

print("Loading text data.")

from sklearn.feature_extraction.text import TfidfVectorizer

# v = TfidfVectorizer(stop_words='english')
# v.fit_transform(word_data, from_data)

            # to_emails.update(to_email)
            # sys.exit(1)

### Store to my_dataset for easy export below.
my_dataset = data_dict

project_email_addresses = list([
    person_data["email_address"] for person_data in data_dict.values() if person_data["email_address"] != "NaN"])

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, ["poi", "email_address"], sort_keys = True)
# labels, features = targetFeatureSplit(data)

email_addresses_train, email_addresses_test = train_test_split(
    project_email_addresses, test_size=0.33, random_state=42)

email_addresses_test = set(email_addresses_test)

print email_addresses_test
print email_addresses_train

with open('final_project_emails.pkl', 'rb') as f:
    project_emails = cPickle.load(f)

non_test_emails = [email for email in project_emails if email_addresses_test.isdisjoint(email["recipients"])]

print "Found %s non-test e-mails (%s)" % (len(non_test_emails), len(non_test_emails) * 100 / float(len(project_emails)))

with open('non_test_emails.pkl', 'wb') as f:
    cPickle.dump(non_test_emails, f)

from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# pipeline = Pipeline([
#     ('scale', StandardScaler()),
#     ('clf', SVC(verbose=False))])
#
# clf = grid_search.GridSearchCV(pipeline, {'clf__C': [1, 10, 100, 1000, 10000]}, n_jobs=8)
# clf = pipeline

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

# dump_classifier_and_data(clf, my_dataset, features_list)