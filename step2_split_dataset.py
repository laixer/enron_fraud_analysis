import cPickle
from sklearn import cross_validation

with open("final_project_dataset_cleaned.pkl", "r") as f:
    data_dict = cPickle.load(f)

data_dict_train, datadict_rest = cross_validation.train_test_split(
    data_dict, test_size=0.33, random_state=42)

data_dict_validation, data_dict_test = cross_validation.train_test_split(
    datadict_rest, test_size=0.5, random_state=42)

with open("final_project_dataset_cleaned_train.pkl", "w") as f:
    cPickle.dump(data_dict_train, f)

with open("final_project_dataset_cleaned_validation.pkl", "w") as f:
    cPickle.dump(data_dict_validation, f)

with open("final_project_dataset_cleaned_test.pkl", "w") as f:
    cPickle.dump(data_dict_test, f)
