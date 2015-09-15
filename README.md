# Goal/process

The goal of this project is to build a model to identify POIs (persons of 
interest) in the context of the Enron fraud scandal. Machine Learning
is useful here as it provides powerful tools to build a model from known
examples and use it to extrapolate to new examples.

I began the process by exporting the data set to a CSV file in order to
load it into R for exploratory analysis. Plotting some of the data in R
quickly revealed some outliers. There was an entry for a person 'TOTAL'
corresponding to the totals row in the financial dataset. Another row
of the data set (for person "BELFER ROBERT" appeared to be incorrectly
shifted column wise). These outliers were removed for the purposes of
building a POI classifier.

After removing outliers, there are 142 entries remaining in the dataset.
The dataset, as expected, is skewed towards non-POIs with only about 13% 
of the people classified POIs. A relatively large proportion (~42%) of
the people in the dataset don't have any data for the e-mail features.
The financial data is mixed. Every person has at least one financial 
feature but individually the amount of values present for a given feature
varies significantly. It's important to note that the lack of a value for 
a particular feature doesn't mean it's missing but rather that the person
didn't have a payment/expense of that type and practically speaking can be
treated as a 0 value.

The financial and e-mail summary metrics didn't look too interesting to 
me so I also tried to build a model using the e-mail text data, but
stopped as it would have required investing a lot more time so the text
based model isn't very good.

# Feature selection

My final model used the following features: exercised_stock_options, 
bonus, total_stock_value, from_poi_to_this_person_pct.

The chosen financial features were chosen to separate people who earned
money from salary vs other income like bonuses and stock value. A higher
bonus or stock grant could be an indicator of a financial reward for 
making money for Enron, which is a possible indicator for fraud that happened
at Enron. Some of the features were selected by seeing their effectiveness
through cross-validation.

I added a new features for the percentage of e-mails sent to/received from 
POIs. My intuition for this was the fact that the absolute numbers could 
vary person to person and a relative number could be a better indicator.

I utilized feature scaling when training an SVM module to avoid any particular
feature getting more importance simply due to its larger range.

# Classifier

I tried Random Forest and SVM classifiers, with the Random Forest doing slightly
better.

# Classifier tuning

Tuning a classifier affects how well or poorly it fits a dataset. It's important
to strike a balance so that a trained classifier doesn't fit too well or too porly
as that would likely make it not extrapolate well to new examples.

I used a grid search to tune the classifiers utilizing the f1 score as the measure
of effectiveness.

# Validation

Validation is the evaluation of a model on a dataset that was not used as part of 
the training to measure its effectiveness on new data. I used cross-validation to 
measure model effectiveness.

# Metrics

| Classifier    | Precision | Recall  |
|---------------|-----------|---------|
| Random Forest | 0.61731   | 0.38150 |
| SVM           | 0.40486	| 0.25000 |

The random classifier was right about 62% percent of the time when identifying
someone as a POI. This is a decent number but probably too low for any serious use.
The SVM classifier didn't fare as well, with only 40% of POI predictions being 
correct.

The recall for both classifiers was relatively low only properly identifying 
38% and 25% of known POIs, respectively. Given new examples, these classifiers
are likely to miss identifying many POIs.