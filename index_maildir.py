"""Generates a single pickle file from a maildir.

Reads all the messages from a maildir and generates a single pickle file
with all the e-mail and recipient information to make it easier to load
and process.
"""

import email
import os
import cPickle

input_maildir = "../maildir"
output_datafile = "maildir.pkl"

maildir_data = []
total_messages = 0
email_files = []
for root, dirnames, filenames in os.walk(input_maildir):
    for filename in filenames:
        email_files.append(os.path.join(root, filename))

for i, filename in enumerate(email_files):
    with open(filename, "r") as f:
        print "Processing [%-80s] (%6s/%6s  %3.0f%%)" % (
            filename, i + 1, len(email_files), (((i + 1) * 100) / float(len(email_files))))

        message = email.message_from_file(f)
        if message.is_multipart():
            raise RuntimeError("didn't expect multipart message")
        text = message.get_payload()
        from_email = message["From"]
        to_email_headers = message.get_all("To", [])
        to_emails = set()
        for _, addr in email.utils.getaddresses(to_email_headers):
            to_emails.add(addr)
        cc_email_headers = message.get_all("Cc", [])
        cc_emails = set()
        for _, addr in email.utils.getaddresses(cc_email_headers):
            cc_emails.add(addr)
        bcc_email_headers = message.get_all("Bcc", [])
        bcc_emails = set()
        for _, addr in email.utils.getaddresses(bcc_email_headers):
            bcc_emails.add(addr)
        # For convenience.
        all_recipients = set(list(to_emails) + list(cc_emails) + list(bcc_emails) + [from_email])
        maildir_data.append({
            'filename': filename,
            'from': from_email,
            'to': to_emails,
            'cc': cc_emails,
            'bcc': bcc_emails,
            'recipients': all_recipients,
            'text': text,
        })

with open(output_datafile, "wb") as f:
    cPickle.dump(maildir_data, f)
