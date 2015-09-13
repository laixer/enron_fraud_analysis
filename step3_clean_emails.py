import cPickle
from collections import defaultdict

print "Loading maildir data."

with open("maildir.pkl", "rb") as f:
    maildir_data = cPickle.load(f)

print "Loading persons data."

with open("final_project_dataset_cleaned.pkl", "r") as f:
    persons = cPickle.load(f)

persons_email_addresses = set([person["email_address"] for person in persons])

relevant_emails = [email for email in maildir_data
                   if not email["recipients"].isdisjoint(persons_email_addresses)]

print "Writing out cleaned maildir data (%d/%d)." % (
    len(relevant_emails), len(maildir_data))

with open("maildir_cleaned.pkl", "wb") as f:
    cPickle.dump(relevant_emails, f)


print "Writing out unique recipients."
recipient_counts = defaultdict(int)
sender_counts = defaultdict(int)
for email in maildir_data:
    recipients = list(email["to"]) + list(email["cc"]) + list(email["bcc"])
    for recipient in recipients:
        recipient_counts[recipient] += 1
    sender_counts[email["from"]] += 1

with open("recipient_counts.txt", "w") as f:
    for recipient, count in recipient_counts.iteritems():
        f.write("%s,%s\n" % (recipient, count))

with open("sender_counts.txt", "w") as f:
    for sender, count in sender_counts.iteritems():
        f.write("%s,%s\n" % (sender, count))

