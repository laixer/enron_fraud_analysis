#!/usr/bin/python
import string

import sys
import pickle
from sklearn import grid_search
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
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
del data_dict['TOTAL']

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
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
pipeline = Pipeline([
    ('scale', StandardScaler()),
    ('clf', SVC(verbose=False))])

clf = grid_search.GridSearchCV(pipeline, {'clf__C': [1, 10, 100, 1000, 10000]}, n_jobs=8)
# clf = pipeline

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)