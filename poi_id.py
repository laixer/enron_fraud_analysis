#!/usr/bin/python
import logging
import string
import sys
import pickle
import pprint
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', level=logging.INFO)


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
        # Divide the data into 10 buckets depending on percentage of emails.
        from_this_person_to_poi_pct = str(int(round((float(from_this_person_to_poi) / float(from_messages)) * 10)))

    from_poi_to_this_person = data_dict[k]["from_poi_to_this_person"]
    to_messages = data_dict[k]["to_messages"]
    if from_poi_to_this_person == 'NaN' or to_messages == 'NaN':
        from_poi_to_this_person_pct = 'NaN'
    else:
        # Divide the data into 10 buckets depending on percentage of emails.
        from_poi_to_this_person_pct = str(int(round((float(from_poi_to_this_person) / float(to_messages)) * 10)))

    data_dict[k]["from_this_person_to_poi_pct"] = from_this_person_to_poi_pct
    data_dict[k]["from_poi_to_this_person_pct"] = from_poi_to_this_person_pct
    if from_messages != 'NaN' and to_messages != 'NaN':
        data_dict[k]["message_ratio"] = float(from_messages) / float(to_messages)
    else:
        data_dict[k]["message_ratio"] = 'NaN'

    if data_dict[k]["salary"] != 'NaN':
        data_dict[k]["salary_pct"] = float(data_dict[k]["salary"]) / (float(data_dict[k]["total_payments"]))
    else:
        data_dict[k]["salary_pct"] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

all_feature_names = data_dict["BAXTER JOHN C"].keys()
all_feature_names.remove("email_address")
# make sure poi is the first feature as that's what the provided code assumes
all_feature_names.remove("poi")
all_feature_names.insert(0, "poi")
all_features_data = featureFormat(my_dataset, all_feature_names, sort_keys=True)
all_features_labels, all_features_features = targetFeatureSplit(all_features_data)
all_feature_names.remove("poi")


def find_best_features(
        feature_names, features, labels, classifier_fun, search_grid,
        normalize_data=False):
    results = []

    processed_features = np.array(features)
    processed_labels = labels
    if normalize_data:
        scaler = StandardScaler()
        processed_features = scaler.fit_transform(
            processed_features, processed_labels)

    feature_selector = SelectKBest(k="all")
    feature_selector.fit(processed_features, processed_labels)

    ranked_features = sorted(zip(feature_names, feature_selector.scores_),
                             key=lambda t: t[1],
                             reverse=True)
    ranked_feature_names = [t[0] for t in ranked_features]

    logging.info("Scored features:\n%s", pprint.pformat(ranked_features))
    logging.info("Ranked feature names: %s", ranked_feature_names)

    for k in range(1, len(feature_names) + 1):
        logging.info("Selecting %s best feature(s)", k)

        selected_feature_names = ranked_feature_names[:k]

        logging.info("Selected features: %s", selected_feature_names)

        feature_indices = [
            feature_names.index(f) for f in selected_feature_names]
        feature_subset = processed_features[:, feature_indices]

        clf = classifier_fun(random_state=98123)

        logging.info("Tuning classifier parameters.")
        clf_tune = grid_search.GridSearchCV(
            clf,
            search_grid,
            n_jobs=-1,
            cv=StratifiedShuffleSplit(labels, n_iter=1000, random_state=42),
            scoring='f1')
        clf_tune.fit(feature_subset, processed_labels)

        logging.info("Scores:\n%s", pprint.pformat(clf_tune.grid_scores_))
        logging.info("Best parameters: %s with score %s",
                     clf_tune.best_params_, clf_tune.best_score_)

        clf = classifier_fun(random_state=1987341, **clf_tune.best_params_)

        logging.info("Testing classifier.")

        precision, recall, f1 = test_classifier(clf, feature_subset, processed_labels)

        results.append((k, precision, recall, f1))

    logging.info("Best features:\n%s", pprint.pformat(results))


def find_best_features_brute_force(
        feature_names, features, labels, classifier_fun, classifier_params,
        search_grid, normalize_data=False):
    selected_feature_names = []
    remaining_features = list(feature_names)

    processed_features = np.array(features)
    processed_labels = labels
    if normalize_data:
        scaler = StandardScaler()
        processed_features = scaler.fit_transform(
            processed_features, processed_labels)

    while remaining_features:
        logging.info("Features selected so far: %s" % selected_feature_names)

        try_features = list(selected_feature_names)
        feature_scores = []
        for pos, try_feature in enumerate(remaining_features):
            logging.info("Trying out [%s] (%s/%s)" % (
                try_feature, pos + 1, len(remaining_features)))

            feature_indices = [
                feature_names.index(f) for f in try_features + [try_feature]]
            feature_subset = processed_features[:, feature_indices]

            clf = classifier_fun(random_state=947619, **classifier_params)

            try:
                logging.info("Tuning classifier parameters.")
                clf_tune = grid_search.GridSearchCV(
                    clf,
                    search_grid,
                    n_jobs=-1,
                    cv=StratifiedShuffleSplit(
                        processed_labels, n_iter=1000, random_state=42),
                    scoring='f1')

                clf_tune.fit(feature_subset, processed_labels)

                logging.info("Scores:\n%s" % pprint.pformat(clf_tune.grid_scores_))
                logging.info("Best parameters: %s with score %s" % (
                    clf_tune.best_params_, clf_tune.best_score_))
                feature_scores.append((try_feature, clf_tune.best_score_, clf_tune.best_params_))

                all_classifier_params = classifier_params.copy()
                all_classifier_params.update(clf_tune.best_params_)
                clf = classifier_fun(random_state=947619, **all_classifier_params)

                logging.info("Testing classifier.")

                test_precision, test_recall, test_f1 = (
                    test_classifier(clf, feature_subset, processed_labels))

                logging.info(
                    "Test score:\nPrecision: %s Recall: %s F1: %s",
                    test_precision, test_recall, test_f1)

            except Exception as e:
                logging.warn("Fit failed: %s" % e)
                feature_scores.append((try_feature, -1, {}))

        feature_scores = sorted(feature_scores, key=lambda t: t[1], reverse=True)

        logging.info("Feature scores: %s" % feature_scores)

        selected_feature = feature_scores[0]
        selected_feature_name = selected_feature[0]
        selected_feature_score = selected_feature[1]
        selected_feature_params = selected_feature[2]

        logging.info("Selected feature: %s" % selected_feature_name)
        selected_feature_names.append(selected_feature_name)
        remaining_features.remove(selected_feature_name)

        feature_indices = [feature_names.index(f) for f in selected_feature_names]
        feature_subset = processed_features[:, feature_indices]

        logging.info("Best parameters: %s with score %s" % (
            selected_feature_params, selected_feature_score))

        all_classifier_params = classifier_params.copy()
        all_classifier_params.update(selected_feature_params)
        clf = classifier_fun(random_state=947619, **all_classifier_params)

        logging.info("Testing classifier.")

        test_classifier(clf, feature_subset, processed_labels)

# logging.info("Finding best features for SVM classifier.")
#
# find_best_features(
#     all_feature_names,
#     all_features_features,
#     all_features_labels,
#     SVC,
#     {'gamma': [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
#      'C': [100, 1000, 10000]},
#     normalize_data=True)
#
# logging.info("Finding best features for random forest classifier.")
#
# find_best_features(
#     all_feature_names,
#     all_features_features,
#     all_features_labels,
#     RandomForestClassifier,
#     {'max_depth': [None, 10],
#      'n_estimators': [10, 100, 1000]})

# logging.info("Finding best features using brute force classification with SVM classifier.")
#
# find_best_features_brute_force(
#     all_feature_names,
#     all_features_features,
#     all_features_labels,
#     SVC,
#     {},
#     {'C': [100, 1000, 10000],
#      'gamma': [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]},
#     normalize_data=True)
#
# logging.info("Finding best features using brute force classification with random forest classifier.")
#
# find_best_features_brute_force(
#     all_feature_names,
#     all_features_features,
#     all_features_labels,
#     RandomForestClassifier,
#     {'n_jobs': -1, 'n_estimators': 100},
#     {})

# find_best_features_svm_custom(all_features)

features_list = [
    'poi',
    'bonus',
    'expenses',
    'message_ratio',
    'from_messages',
    'director_fees',
    'salary_pct',
]

scaler = StandardScaler()
processed_features = scaler.fit_transform(all_features_features, all_features_labels)

clf = SVC(verbose=False, random_state=947619, C=100, gamma=2.0)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

print("Testing classifier.")

feature_indices = [all_feature_names.index(f) for f in features_list if f != 'poi']
feature_subset = np.array(processed_features)[:, feature_indices]

test_classifier(clf, feature_subset, all_features_labels)

# SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=2.0,
#     kernel='rbf', max_iter=-1, probability=False, random_state=947619,
#     shrinking=True, tol=0.001, verbose=False)
# Accuracy: 0.92607	Precision: 0.82781	Recall: 0.56250	F1: 0.66984	F2: 0.60103
# Total predictions: 15000	True positives: 1125	False positives:  234	False negatives:  875	True negatives: 12766



#
# ### Dump your classifier, dataset, and features_list so
# ### anyone can run/check your results.

print("Dumping classifier.")

dump_classifier_and_data(clf, my_dataset, features_list)
