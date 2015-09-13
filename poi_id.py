#!/usr/bin/python
import string
import sys
import pickle
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search, cross_validation

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
# pipeline = Pipeline([
#     ('scale', StandardScaler()),
#     ('clf', SVC(verbose=False))])

# print("Tuning classifier parameters")
# data = featureFormat(my_dataset, features_list, sort_keys=True)
# labels, features = targetFeatureSplit(data)
#
#
# rf_tune = grid_search.GridSearchCV(rf, {'n_estimators': [10, 100, 200, 500, 1000],
#                                         'max_depth': [None, 10, 100, 1000, 2000, 5000]},
#                                    n_jobs=-1,
#                                    cv=StratifiedShuffleSplit(labels, n_iter=1000, random_state = 42))
#
#
#
# rf_tune.fit(features, labels)
#
# print rf_tune.grid_scores_
# print rf_tune.best_estimator_
# print rf_tune.best_score_
# print rf_tune.best_params_

clf = RandomForestClassifier(n_estimators=100, max_depth=1000, random_state=1987341)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print("Testing classifier")

test_classifier(clf, my_dataset, features_list)
#
# ### Dump your classifier, dataset, and features_list so
# ### anyone can run/check your results.
#
print("Dumping classifier")

dump_classifier_and_data(clf, my_dataset, features_list)
