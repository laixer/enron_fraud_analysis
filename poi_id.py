#!/usr/bin/python
import string
import sys
import pickle
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit
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

# Export dataset in CSV format to make it easier to load into other tools (R, etc)
write_data_dict_to_csv("final_project_dataset.csv")

### Task 2: Remove outliers
del data_dict['TOTAL']  # Summary row from financial data
del data_dict['BELFER ROBERT']  # Data incorrectly imported

### Task 3: Create new feature(s)
for k, v in data_dict.iteritems():
    from_this_person_to_poi = data_dict[k]["from_this_person_to_poi"]
    from_messages = data_dict[k]["from_messages"]

    if from_this_person_to_poi == 'NaN' or from_messages == 'NaN':
        from_this_person_to_poi_pct = 'NaN'
    else:
        from_this_person_to_poi_pct = str(int(round((float(from_this_person_to_poi) / float(from_messages)) * 10)))

    from_poi_to_this_person = data_dict[k]["from_poi_to_this_person"]
    to_messages = data_dict[k]["to_messages"]
    if from_poi_to_this_person == 'NaN' or to_messages == 'NaN':
        from_poi_to_this_person_pct = 'NaN'
    else:
        from_poi_to_this_person_pct = str(int(round((float(from_poi_to_this_person) / float(to_messages)) * 10)))

    # print from_this_person_to_poi
    # print from_messages
    # print from_this_person_to_poi_pct
    data_dict[k]["from_this_person_to_poi_pct"] = from_this_person_to_poi_pct
    data_dict[k]["from_poi_to_this_person_pct"] = from_poi_to_this_person_pct
    if from_messages != 'NaN' and to_messages != 'NaN':
        data_dict[k]["message_ratio"] = float(from_messages) / float(to_messages)
    else:
        data_dict[k]["message_ratio"] = 'NaN'
    if data_dict[k]["bonus"] != 'NaN':
        total_stock_value = data_dict[k]["total_stock_value"]
        if total_stock_value != 'NaN':
            total_stock_value = float(total_stock_value)
        else:
            total_stock_value = 0
        data_dict[k]["salary_pct"] = float(data_dict[k]["bonus"]) / (float(data_dict[k]["total_payments"]))
    else:
        data_dict[k]["salary_pct"] = 'NaN'

features_list = [
    'poi',
    'exercised_stock_options',
    'bonus',
    'total_stock_value',
    # 'shared_receipt_with_poi',
    # 'from_messages',
    # 'to_messages',
    # 'from_this_person_to_poi',
    # 'from_this_person_to_poi_pct',
    # 'from_poi_to_this_person',
    'from_poi_to_this_person_pct',
    # 'message_ratio',
    # 'salary_pct',
]

matches = []

### Store to my_dataset for easy export below.
my_dataset = data_dict

rf = RandomForestClassifier()
svm = Pipeline([
    ('scale', StandardScaler()),
    ('clf', SVC(verbose=False, random_state=947619))])

print("Tuning classifier parameters.")
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
# rf_tune = grid_search.GridSearchCV(rf,
#                                    {'n_estimators': [10, 100, 200, 500, 1000],
#                                     'max_depth': [None, 10, 100, 1000, 2000, 5000]},
#                                    n_jobs=-1,
#                                    cv=StratifiedShuffleSplit(labels, n_iter=1000, random_state=42),
#                                    scoring='f1')
# rf_tune.fit(features, labels)
# print rf_tune.grid_scores_
# print rf_tune.best_params_
# print rf_tune.best_score_

#
# [mean: 0.36641, std: 0.33340, params: {'n_estimators': 10, 'max_depth': None},
#  mean: 0.43272, std: 0.33645, params: {'n_estimators': 100, 'max_depth': None},
#  mean: 0.42885, std: 0.33737, params: {'n_estimators': 200, 'max_depth': None},
#  mean: 0.42587, std: 0.33677, params: {'n_estimators': 500, 'max_depth': None},
#  mean: 0.42384, std: 0.33755, params: {'n_estimators': 1000, 'max_depth': None},
#  mean: 0.37597, std: 0.34234, params: {'n_estimators': 10, 'max_depth': 10},
#  mean: 0.42831, std: 0.33774, params: {'n_estimators': 100, 'max_depth': 10},
#  mean: 0.43099, std: 0.33541, params: {'n_estimators': 200, 'max_depth': 10},
#  mean: 0.42609, std: 0.33392, params: {'n_estimators': 500, 'max_depth': 10},
#  mean: 0.42517, std: 0.33794, params: {'n_estimators': 1000, 'max_depth': 10},
#  mean: 0.38634, std: 0.34296, params: {'n_estimators': 10, 'max_depth': 100},
#  mean: 0.43465, std: 0.33633, params: {'n_estimators': 100, 'max_depth': 100},
#  mean: 0.42265, std: 0.33734, params: {'n_estimators': 200, 'max_depth': 100},
#  mean: 0.42224, std: 0.33868, params: {'n_estimators': 500, 'max_depth': 100},
#  mean: 0.42607, std: 0.33633, params: {'n_estimators': 1000, 'max_depth': 100},
#  mean: 0.36761, std: 0.34362, params: {'n_estimators': 10, 'max_depth': 1000},
#  mean: 0.42452, std: 0.33502, params: {'n_estimators': 100, 'max_depth': 1000},
#  mean: 0.42589, std: 0.33873, params: {'n_estimators': 200, 'max_depth': 1000},
#  mean: 0.42592, std: 0.33670, params: {'n_estimators': 500, 'max_depth': 1000},
#  mean: 0.42198, std: 0.33650, params: {'n_estimators': 1000, 'max_depth': 1000},
#  mean: 0.36641, std: 0.33848, params: {'n_estimators': 10, 'max_depth': 2000},
#  mean: 0.42347, std: 0.33622, params: {'n_estimators': 100, 'max_depth': 2000},
#  mean: 0.43067, std: 0.33836, params: {'n_estimators': 200, 'max_depth': 2000},
#  mean: 0.42494, std: 0.33785, params: {'n_estimators': 500, 'max_depth': 2000},
#  mean: 0.42462, std: 0.33635, params: {'n_estimators': 1000, 'max_depth': 2000},
#  mean: 0.36407, std: 0.33110, params: {'n_estimators': 10, 'max_depth': 5000},
#  mean: 0.42293, std: 0.33776, params: {'n_estimators': 100, 'max_depth': 5000},
#  mean: 0.42977, std: 0.34011, params: {'n_estimators': 200, 'max_depth': 5000},
#  mean: 0.42854, std: 0.33870, params: {'n_estimators': 500, 'max_depth': 5000},
#  mean: 0.42402, std: 0.33688, params: {'n_estimators': 1000, 'max_depth': 5000}]
# {'n_estimators': 100, 'max_depth': 100}
# 0.43465


# svm_tune = grid_search.GridSearchCV(svm, {'clf__C': [10, 100, 1000, 10000],
#                                           'clf__gamma': [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]},
#                                     n_jobs=-1,
#                                     cv=StratifiedShuffleSplit(labels, n_iter=1000, random_state=42),
#                                     scoring='f1')
# svm_tune.fit(features, labels)
# print svm_tune.grid_scores_
# print svm_tune.best_params_
# print svm_tune.best_score_

# [mean: 0.15630, std: 0.28541, params: {'clf__gamma': 0.01, 'clf__C': 10},
#  mean: 0.12793, std: 0.25452, params: {'clf__gamma': 0.1, 'clf__C': 10},
#  mean: 0.16422, std: 0.27627, params: {'clf__gamma': 0.2, 'clf__C': 10},
#  mean: 0.16534, std: 0.27160, params: {'clf__gamma': 0.5, 'clf__C': 10},
#  mean: 0.16569, std: 0.27295, params: {'clf__gamma': 1.0, 'clf__C': 10},
#  mean: 0.28249, std: 0.33831, params: {'clf__gamma': 2.0, 'clf__C': 10},
#  mean: 0.24824, std: 0.32079, params: {'clf__gamma': 5.0, 'clf__C': 10},
#  mean: 0.25354, std: 0.32528, params: {'clf__gamma': 10.0, 'clf__C': 10},
#  mean: 0.16033, std: 0.28366, params: {'clf__gamma': 0.01, 'clf__C': 100},
#  mean: 0.27960, std: 0.32179, params: {'clf__gamma': 0.1, 'clf__C': 100},
#  mean: 0.24442, std: 0.30156, params: {'clf__gamma': 0.2, 'clf__C': 100},
#  mean: 0.24000, std: 0.30570, params: {'clf__gamma': 0.5, 'clf__C': 100},
#  mean: 0.24521, std: 0.31351, params: {'clf__gamma': 1.0, 'clf__C': 100},
#  mean: 0.28082, std: 0.31975, params: {'clf__gamma': 2.0, 'clf__C': 100},
#  mean: 0.29710, std: 0.32129, params: {'clf__gamma': 5.0, 'clf__C': 100},
#  mean: 0.29080, std: 0.32878, params: {'clf__gamma': 10.0, 'clf__C': 100},
#  mean: 0.17114, std: 0.28575, params: {'clf__gamma': 0.01, 'clf__C': 1000},
#  mean: 0.24267, std: 0.29899, params: {'clf__gamma': 0.1, 'clf__C': 1000},
#  mean: 0.25682, std: 0.30849, params: {'clf__gamma': 0.2, 'clf__C': 1000},
#  mean: 0.25864, std: 0.30673, params: {'clf__gamma': 0.5, 'clf__C': 1000},
#  mean: 0.30284, std: 0.31334, params: {'clf__gamma': 1.0, 'clf__C': 1000},
#  mean: 0.28319, std: 0.31407, params: {'clf__gamma': 2.0, 'clf__C': 1000},
#  mean: 0.29740, std: 0.32162, params: {'clf__gamma': 5.0, 'clf__C': 1000},
#  mean: 0.29080, std: 0.32878, params: {'clf__gamma': 10.0, 'clf__C': 1000},
#  mean: 0.25724, std: 0.32063, params: {'clf__gamma': 0.01, 'clf__C': 10000},
#  mean: 0.29804, std: 0.31280, params: {'clf__gamma': 0.1, 'clf__C': 10000},
#  mean: 0.28182, std: 0.31062, params: {'clf__gamma': 0.2, 'clf__C': 10000},
#  mean: 0.28841, std: 0.31121, params: {'clf__gamma': 0.5, 'clf__C': 10000},
#  mean: 0.26640, std: 0.30053, params: {'clf__gamma': 1.0, 'clf__C': 10000},
#  mean: 0.28319, std: 0.31407, params: {'clf__gamma': 2.0, 'clf__C': 10000},
#  mean: 0.29740, std: 0.32162, params: {'clf__gamma': 5.0, 'clf__C': 10000},
#  mean: 0.29080, std: 0.32878, params: {'clf__gamma': 10.0, 'clf__C': 10000}]
# {'clf__gamma': 1.0, 'clf__C': 1000}
# 0.30283968254

clf = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=1987341)
# clf = Pipeline([
#     ('scale', StandardScaler()),
#     ('clf', SVC(verbose=False, random_state=947619, C=1000))])

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print("Testing classifier.")

test_classifier(clf, my_dataset, features_list)

# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=100, max_features='auto', max_leaf_nodes=None,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#                        oob_score=False, random_state=1987341, verbose=0,
#                        warm_start=False)
# Accuracy: 0.86846	Precision: 0.61731	Recall: 0.38150	F1: 0.47157	F2: 0.41306
# Total predictions: 13000	True positives:  763	False positives:  473	False negatives: 1237	True negatives: 10527


# Pipeline(steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)),
#                  ('clf', SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
#                              kernel='rbf', max_iter=-1, probability=False, random_state=947619,
#                              shrinking=True, tol=0.001, verbose=False))])
# Accuracy: 0.82808	Precision: 0.40486	Recall: 0.25000	F1: 0.30912	F2: 0.27071
# Total predictions: 13000	True positives:  500	False positives:  735	False negatives: 1500	True negatives: 10265


#
# ### Dump your classifier, dataset, and features_list so
# ### anyone can run/check your results.

print("Dumping classifier.")

dump_classifier_and_data(clf, my_dataset, features_list)
