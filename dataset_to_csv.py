"""Exports persons dataset as a CSV file for analyzing in R.

Reads final_project_dataset.pkl and saves the data to final_project_dataset.csv
"""

import cPickle
import string

def write_data_dict_to_csv(data_dict, output_file):
    all_features = data_dict["BAXTER JOHN C"].keys()
    with open(output_file, "w") as f:
        f.write("name," + string.join(all_features, ",") + "\n")
        for person, person_data in data_dict.iteritems():
            person_data_values = list(map(lambda feature: str(person_data[feature]), all_features))
            f.write(person + "," + string.join(person_data_values, ",") + "\n")

with open("final_project_dataset.pkl", "r") as f:
    data_dict = cPickle.load(f)

write_data_dict_to_csv(data_dict, "final_project_dataset.csv")
