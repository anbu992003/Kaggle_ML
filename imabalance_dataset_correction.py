
XGBoost Parameter Tuning
=================================================================================================================================================
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python//
seed
eval_metric -  logloss (or) error(default)
objective - 'binary:logistic' (multi:softprob)
scale_pos_weight - mentioned below
subsample - 05.1
colsample_bytree - 0.5-1
max_delta_step >0
max_depth >10
max_leaf_nodes >2^n where n=max_depth

Call predict_proba instead of binary:logistic



https://datascience.stackexchange.com/questions/17857/xgboost-for-binary-classification-choosing-the-right-threshold
If you want to maximize f1 metric, one approach is to train your classifier to predict a probability, then choose a threshold that maximizes the f1 score. The threshold probably won't be 0.5.



XGBoost data imbalance
=================================================================================================================================================
https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets

Generally, scale_pos_weight is the ratio of number of negative class to the positive class.

Suppose, the dataset has 90 observations of negative class and 10 observations of positive class, then ideal value of scale_pos_weight should be 9.

See the doc: http://xgboost.readthedocs.io/en/latest/parameter.html
scale_pos_weight = count(negative examples)/count(Positive examples)
scale_pos_weight = sqrt(count(negative examples)/count(Positive examples)) 



https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
Handle Imbalanced Dataset
For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of XGBoost model, and there are two ways to improve it.

If you care only about the overall performance metric (AUC) of your prediction

Balance the positive and negative weights via scale_pos_weight

Use AUC for evaluation

If you care about predicting the right probability

In such a case, you cannot re-balance the dataset

Set parameter max_delta_step to a finite number (say 1) to help convergence






General Imbalance correction
=================================================================================================================================================
https://towardsdatascience.com/what-to-do-when-your-classification-dataset-is-imbalanced-6af031b12a36


##############UpSample Data
X = df.drop(‘diagnosis’,axis=1)
y = df[‘diagnosis’]
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
#split data into test and training sets
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
#combine them back for resampling
train_data = pd.concat([X_train, y_train], axis=1)
# separate minority and majority classes
negative = train_data[train_data.diagnosis==0]
positive = train_data[train_data.diagnosis==1]
# upsample minority
pos_upsampled = resample(positive,
 replace=True, # sample with replacement
 n_samples=len(negative), # match number in majority class
 random_state=27) # reproducible results
# combine majority and upsampled minority
upsampled = pd.concat([negative, pos_upsampled])
# check new class counts
upsampled.diagnosis.value_counts()
1    139
0    139
Name: diagnosis, dtype: int64


############Downsample DAta

# downsample majority
neg_downsampled = resample(negative,
 replace=True, # sample with replacement
 n_samples=len(positive), # match number in minority class
 random_state=27) # reproducible results
# combine minority and downsampled majority
downsampled = pd.concat([positive, neg_downsampled])
# check new class counts
downsampled.diagnosis.value_counts()
1    41
0    41
Name: diagnosis, dtype: int64




############Synthetic DAta

from imblearn.over_sampling import SMOTE
# Separate input features and target
X = df.drop(‘diagnosis’,axis=1)
y = df[‘diagnosis’]
# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
sm = SMOTE(random_state=27, ratio=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)
X_train.shape, y_train.shape
((314, 5), (314,)) #We now have 314 data items in our training set
y_train = pd.DataFrame(y_train, columns = ['diagnosis'])
y_train.diagnosis.value_counts()
1    157
0    157
Name: diagnosis, dtype: int64