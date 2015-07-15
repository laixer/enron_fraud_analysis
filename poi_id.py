#!/usr/bin/python
import email
import os
from pprint import pprint
import string

import sys
import pickle
import cPickle
from sklearn import grid_search
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

from email.parser import Parser

matches = []

poi_email_addresses = set()
for name, data in data_dict.iteritems():
    if data['poi']:
        poi_email_addresses.add(data['email_address'])

print("Loading text data.")

def process_maildir(maildir):
    email_data = []
    total_messages = 0
    any_poi_messages = 0
    all_poi_messages = 0
    email_files = []
    for root, dirnames, filenames in os.walk(maildir):
        for filename in filenames:
            email_files.append(os.path.join(root, filename))

    stemmer = SnowballStemmer("english")

    for i, filename in enumerate(email_files):
        with open(filename, "r") as f:
            print "Processing [%-80s] (%6s/%6s  %3.0f%%)" % (
                filename, i + 1, len(email_files), (((i + 1) * 100) / float(len(email_files))))
            message = email_parser.parse(f)
            if message.is_multipart():
                raise RuntimeError("didn't expect multipart message")
            text = message.get_payload()
            from_email = message["From"]
            to_email_headers = message.get_all("To", [])
            to_emails = set()
            for _, addr in email.utils.getaddresses(to_email_headers):
                to_emails.add(addr)
            cc_email_headers = message.get_all("Cc", [])
            cc_emails = set()
            for _, addr in email.utils.getaddresses(cc_email_headers):
                cc_emails.add(addr)
            bcc_email_headers = message.get_all("Bcc", [])
            bcc_emails = set()
            for _, addr in email.utils.getaddresses(bcc_email_headers):
                bcc_emails.add(addr)
            all_recipients = set(list(to_emails) + list(cc_emails) + list(bcc_emails) + [from_email])
            total_messages += 1
            any_poi = any(r in poi_email_addresses for r in all_recipients)
            if any_poi:
                any_poi_messages += 1
            all_poi = all(r in poi_email_addresses for r in all_recipients)
            if all_poi:
                all_poi_messages += 1

            stemmed_words = string.join(map(stemmer.stem, text.split()))

            email_data.append({
                'filename': filename,
                'any_poi': any_poi,
                'all_poi': all_poi,
                'from': from_email,
                'to': to_emails,
                'cc': cc_emails,
                'bcc': bcc_emails,
                'recipients': all_recipients,
                'text': text,
                'stemmed_words': stemmed_words,
            })

    print("Total: %s, all poi %s, any poi %s" %(total_messages, all_poi_messages, any_poi_messages))

    return email_data

# email_data = process_maildir('../maildir')
# print("Writing out data.")
# with open('enron_email_data.pkl', 'wb') as f:
#     cPickle.dump(email_data, f)

with open('enron_email_data.pkl', 'rb') as f:
    email_data = cPickle.load(f)

print len(email_data)

from sklearn.feature_extraction.text import TfidfVectorizer

# v = TfidfVectorizer(stop_words='english')
# v.fit_transform(word_data, from_data)

            # to_emails.update(to_email)
            # sys.exit(1)

# read emails 1m7s

    #     for name, data in data_dict.iteritems():
    # print data['email_address']

# just salary:
# GaussianNB()
# 	Accuracy: 0.79660	Precision: 0.46398	Recall: 0.10950	F1: 0.17718	F2: 0.12925
# 	Total predictions: 10000	True positives:  219	False positives:  253	False negatives: 1781	True negatives: 7747
#
# 'poi', 'deferral_payments', 'exercised_stock_options', 'bonus', 'total_stock_value', 'deferred_income',
# 'from_this_person_to_poi_pct'
# GaussianNB()
# 	Accuracy: 0.85079	Precision: 0.46851	Recall: 0.33100	F1: 0.38793	F2: 0.35164
# 	Total predictions: 14000	True positives:  662	False positives:  751	False negatives: 1338	True negatives: 11249

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

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