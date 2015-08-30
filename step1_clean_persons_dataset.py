"""Removes outliers from the persons dataset.

Reads final_project_dataset.pkl and saves cleaned dataset to
final_project_dataset_cleaned.pkl
"""

import cPickle

with open("final_project_dataset.pkl", "r") as f:
    data_dict = cPickle.load(f)

del data_dict['TOTAL']  # Summary row from financial data
del data_dict['BELFER ROBERT']  # Data incorrectly imported

with open("final_project_dataset_cleaned.pkl", "w") as f:
    cPickle.dump(data_dict, f)
