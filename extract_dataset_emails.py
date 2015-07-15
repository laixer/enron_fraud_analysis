import cPickle
import string
from nltk import SnowballStemmer

maildir_input_file = 'maildir.pkl'
project_dataset_file = "final_project_dataset.pkl"
project_emails_output_file = "final_project_emails.pkl"

print "Loading project data set."

with open(project_dataset_file, 'rb') as f:
    project_dataset = cPickle.load(f)

print "Loading maildir data."

with open(maildir_input_file, 'rb') as f:
    maildir_data = cPickle.load(f)

project_email_addresses = set([person_data["email_address"] for person_data in project_dataset.values()])
poi_email_addresses = set([person_data["email_address"] for person_data in project_dataset.values() if person_data["poi"]])

print "Filtering and processing emails."

stemmer = SnowballStemmer("english")

any_poi_messages = 0
all_poi_messages = 0
project_emails = list([email for email in maildir_data if not project_email_addresses.isdisjoint(email["recipients"])])
for email in project_emails:
    any_poi = any(r in poi_email_addresses for r in email["recipients"])
    all_poi = all(r in poi_email_addresses for r in email["recipients"])
    email["stemmed_words"] = string.join(map(stemmer.stem, email["text"].split()))
    if any_poi:
        any_poi_messages += 1
    if all_poi:
        all_poi_messages += 1

print "Writing out %s emails, any poi %s, all poi %s" % (len(project_emails), any_poi_messages, all_poi_messages)

with open(project_emails_output_file, 'wb') as f:
    cPickle.dump(project_emails, f)

# 359MB


