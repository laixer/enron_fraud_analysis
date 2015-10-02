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

My final model used the following features: bonus, 
expenses, message_ratio, from_messages, director_fees, salary_pct.

I added a new features for the percentage of e-mails sent to/received from 
POIs. My intuition for this was the fact that the absolute numbers could 
vary person to person and a relative number could be a better indicator.

I utilized feature scaling when training an SVM module to avoid any particular
feature getting more importance simply due to its larger range.

I used the SelectKBest algorithm with the default ANOVA F-value scorer.
For fun I also tried a slightly less scientific approach by incrementally adding
features that yielded the best F1 score. 

## SelectKBest using SVM classifier

| +feature                    | precision | recall |
|-----------------------------|-----------|--------|
| exercised_stock_options     | 1.0000    | 0.2170 |
| total_stock_value           | 0.9644    | 0.2170 |
| bonus                       | 0.4784    | 0.3385 |
| salary                      | 0.4342    | 0.2705 |
| from_this_person_to_poi_pct | 0.2247    | 0.2390 |
| deferred_income             | 0.2110    | 0.1860 |
| long_term_incentive         | 0.2170    | 0.1580 |
| restricted_stock            | 0.2454    | 0.2220 |
| total_payments              | 0.2068    | 0.1695 |
| shared_receipt_with_poi     | 0.2328    | 0.2455 |
| loan_advances               | 0.2326    | 0.2460 |
| expenses                    | 0.2323    | 0.2555 |
| from_poi_to_this_person     | 0.2038    | 0.2165 |
| other                       | 0.2646    | 0.3000 |
| salary_pct                  | 0.2288    | 0.2515 |
| from_this_person_to_poi     | 0.2497    | 0.2465 |
| director_fees               | 0.2505    | 0.2490 |
| to_messages                 | 0.2904    | 0.2640 |
| from_poi_to_this_person_pct | 0.3266    | 0.3310 |
| deferral_payments           | 0.3199    | 0.3300 |
| from_messages               | 0.3406    | 0.3420 |
| message_ratio               | 0.3285    | 0.3295 |
| restricted_stock_deferred   | 0.3320    | 0.3335 |

## SelectKBest using Random Forest classifier

| +feature                    | precision | recall |
|-----------------------------|-----------|--------|
| exercised_stock_options     | 0.3539    | 0.2180 |
| total_stock_value           | 0.3289    | 0.1970 |
| bonus                       | 0.6360    | 0.3670 |
| salary                      | 0.5194    | 0.2740 |
| from_this_person_to_poi_pct | 0.4929    | 0.2245 |
| deferred_income             | 0.4448    | 0.1955 |
| long_term_incentive         | 0.3822    | 0.1395 |
| restricted_stock            | 0.4233    | 0.1215 |
| total_payments              | 0.4630    | 0.1440 |
| shared_receipt_with_poi     | 0.4220    | 0.1610 |
| loan_advances               | 0.3904    | 0.1300 |
| expenses                    | 0.5301    | 0.1675 |
| from_poi_to_this_person     | 0.4241    | 0.1355 |
| other                       | 0.5201    | 0.1680 |
| salary_pct                  | 0.5222    | 0.1585 |
| from_this_person_to_poi     | 0.5049    | 0.1560 |
| director_fees               | 0.5145    | 0.1595 |
| to_messages                 | 0.5076    | 0.1505 |
| from_poi_to_this_person_pct | 0.5180    | 0.1580 |
| deferral_payments           | 0.5246    | 0.1600 |
| from_messages               | 0.4425    | 0.1425 |
| message_ratio               | 0.4257    | 0.1360 |
| restricted_stock_deferred   | 0.5161    | 0.1445 |

## Brute force using SVM classifier

| +feature                    | precision | recall |
|-----------------------------|-----------|--------|
| bonus                       | 0.5588    | 0.2900 |
| expenses                    | 0.6437    | 0.4895 |
| message_ratio               | 0.6253    | 0.6150 |
| from_messages               | 0.6560    | 0.6255 |
| director_fees               | 0.6504    | 0.6250 |
| salary_pct                  | 0.8278    | 0.5625 |
| other                       | 0.7379    | 0.6095 |
| loan_advances               | 0.7366    | 0.6095 |
| restricted_stock_degerred   | 0.7313    | 0.6095 |
| total_payments              | 0.7280    | 0.6075 |
| from_this_person_to_poi     | 0.6996    | 0.5310 |
| deferral_payments           | 0.6950    | 0.5105 |
| deferred_income             | 0.5449    | 0.4580 |
| to_messages                 | 0.6278    | 0.4225 |
| long_term_incentive         | 0.7248    | 0.4175 |
| exercised_stock_options     | 0.6278    | 0.3795 |
| total_stock_value           | 0.5901    | 0.3585 |
| shared_receipt_with_poi     | 0.4224    | 0.4395 |
| salary                      | 0.4307    | 0.4305 |
| from_poi_to_this_person_pct | 0.4285    | 0.4300 |
| from_poi_to_this_person     | 0.4720    | 0.4670 |
| restricted_stock            | 0.4268    | 0.4300 |
| from_this_person_to_poi_pct | 0.3320    | 0.3335 |

## Brute force using Random Forest classifier

| +feature                    | precision | recall |
|-----------------------------|-----------|--------|
| message_ratio               | 0.3736    | 0.3125 |
| from_poi_to_this_person_pct | 0.4063    | 0.3185 |
| exercised_stock_options     | 0.5766    | 0.3030 |
| bonus                       | 0.6086    | 0.3740 |
| loan_advances               | 0.6447    | 0.3710 |
| restricted_stock_deferred   | 0.6625    | 0.3710 |
| director_fees               | 0.6895    | 0.3675 |
| to_messages                 | 0.7078    | 0.3585 |
| deferral_payments           | 0.7032    | 0.3495 |
| total_payments              | 0.6576    | 0.3035 |
| expenses                    | 0.6466    | 0.2845 |
| total_stock_value           | 0.6156    | 0.2490 |
| from_this_person_to_poi_pct | 0.6201    | 0.2530 |
| from_poi_to_this_person     | 0.5946    | 0.2420 |
| deferred_income             | 0.5776    | 0.2290 |
| shared_receipt_with_poi     | 0.6165    | 0.2540 |
| long_term_incentive         | 0.6089    | 0.2390 |
| from_messages               | 0.5699    | 0.2080 |
| from_this_person_to_poi     | 0.5356    | 0.1955 |
| other                       | 0.4920    | 0.1700 |
| salary_pct                  | 0.5429    | 0.1740 |
| restricted_stock            | 0.5237    | 0.1545 |
| salary                      | 0.5054    | 0.1410 |


# Classifier

I tried Random Forest and SVM classifiers, with the SVM classifier performing 
better overall.

# Classifier tuning

Tuning a classifier affects how well or poorly it fits a dataset. It's important
to strike a balance so that a trained classifier doesn't fit too well or too 
poorly as that would likely make it not extrapolate well to new examples.

I used a grid search to tune the classifiers utilizing the f1 score as the 
measure of effectiveness.

# Validation

Validation is the evaluation of a model on a dataset that was not used as part 
of the training to measure its effectiveness on new data. I used 
cross-validation to measure model effectiveness.

# Metrics

| Precision | 0.81797   |
| Recall    | 0.53700	|

The trained SVM classifier was right about 82% percent of the time when 
identifying someone as a POI. This is a decent number if someone wanted to 
identify people worth a closer look, but not high enough to make any 
immediate conclusions.

The recall was about 54%, which means there's a almost a 50% chance that it 
would miss identifying a POI as a POI. Due to the relatively low recall score,
this classifier would not be useful if one was trying to identify as many POIs
as possible.
