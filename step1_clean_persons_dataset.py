"""Removes outliers from the persons dataset.

Reads final_project_dataset.pkl and saves cleaned dataset to
final_project_dataset_cleaned.pkl

Cleaning removes outliers and transforms the dataset from a
dict to an array of dict for easier usage.

Outliers were found by analyzing the dataset in R (see analysis.Rmd)
"""

import cPickle

with open("final_project_dataset.pkl", "r") as f:
    data_dict = cPickle.load(f)

del data_dict['TOTAL']  # Summary row from financial data
del data_dict['BELFER ROBERT']  # Data incorrectly imported

for name, data in data_dict.iteritems():
    data["name"] = name

persons = [person for person in data_dict.values()
           if person["email_address"] != "NaN"]

with open("final_project_dataset_cleaned.pkl", "w") as f:
    cPickle.dump(persons, f)
