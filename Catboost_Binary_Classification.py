

CAtboost
https://github.com/catboost/tutorials

https://catboost.ai/docs/concepts/python-usages-examples.html

1)Binary Classification
cat_features = [0, 1]
train_data = [["a", "b", 1, 4, 5, 6],
              ["a", "b", 4, 5, 6, 7],
              ["c", "d", 30, 40, 50, 60]]
train_labels = [1, 1, -1]
eval_data = [["a", "b", 2, 4, 6, 8],
             ["a", "d", 1, 4, 50, 60]]

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2,
                           learning_rate=1,
                           depth=2)
# Fit model
model.fit(train_data, train_labels, cat_features)
# Get predicted classes
preds_class = model.predict(eval_data)
# Get predicted probabilities for each class
preds_proba = model.predict_proba(eval_data)
# Get predicted RawFormulaVal
preds_raw = model.predict(eval_data, prediction_type='RawFormulaVal')


1)Binary Classification
from catboost import CatBoostClassifier, Pool

train_data = Pool(data=[[1, 4, 5, 6],
                        [4, 5, 6, 7],
                        [30, 40, 50, 60]],
                  label=[1, 1, -1],
                  weight=[0.1, 0.2, 0.3])

model = CatBoostClassifier(iterations=10)

model.fit(train_data)
preds_class = model.predict(train_data)