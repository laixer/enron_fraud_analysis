import cPickle

print "Loading maildir data."

with open("maildir_stemmed.pkl", "rb") as f:
    maildir_data = cPickle.load(f)

print "Loading TF-IDF extractor."

with open("maildir_tf_idf.pkl", "rb") as f:
    tf_idf_extractor = cPickle.load(f)

print "Loading persons dataset."

with open("final_project_dataset_cleaned_validation.pkl", "rb") as f:
    persons_data = cPickle.load(f)

print "Loading email classifier."

with open("email_classifier.pkl", "rb") as f:
    email_classifier = cPickle.load(f)

persons = [person for person in persons_data if person["email_address"] != "NaN"]

for person in persons:
    name = person["name"]
    email_address = person["email_address"]

    print "Finding all e-mails for [%s] [%s]" % (name, email_address)
    emails = [email for email in maildir_data if email_address == email["from"]]
    print "Found %s emails." % len(emails)

    if len(emails):
        features = tf_idf_extractor.transform([email["stemmed_text"] for email in emails])

        labels = email_classifier.predict(features)
        print "# interesting e-mails: %s (is_poi: %s)" % (len(filter(None, labels)), person["poi"])
    # print labels

