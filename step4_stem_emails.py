import cPickle
import string
from nltk import SnowballStemmer

print "Loading maildir data."

with open("maildir_cleaned.pkl", "rb") as f:
    maildir_data = cPickle.load(f)

print "Stemming e-mails."

stemmer = SnowballStemmer("english")

for email in maildir_data:
    email["stemmed_text"] = string.join(map(stemmer.stem, email["text"].split()))

print "Writing out %s emails." % len(maildir_data)

with open("maildir_stemmed.pkl", "wb") as f:
    cPickle.dump(maildir_data, f)

