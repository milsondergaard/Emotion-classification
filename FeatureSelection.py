
# %%
### FEATURE SELECTION
#Cross-validated Feature selection using recursive feature elimination

print(__doc__)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
# load data
# load data
df = pd.read_csv('features.csv')
df = df[df.concreteness.notnull()]
X = df.loc[:, df.columns != 'label']
X = X.drop(['sentence_tokenize', 'sentence_lower'], axis=1)
y = df['label']

#the parameter C controls the sparsity: the smaller C the fewer features selected 
# Create the RFE object and compute a cross-validated score.
#svc = SVC(kernel="linear")
lasso = LassoCV()
# The "accuracy" scoring is proportional to the number of correct
# classifications
model = RFECV(estimator=lasso, step=1, cv=StratifiedKFold(5))
model.fit(X, y)

#inspecting
#support = array of true and false
support = model.support_
# ranking = features chosen = 1
ranking = model.ranking_
#cross validation scores, one scores for each feature
grid_scores = model.grid_scores_
#number of selected features
selected_features = model.n_features_
# same as support, mask of selected features
model.get_support()

#information about the model
model.get_params()
model.set_params()
model.estimator_.coef_

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(model.grid_scores_) + 1), model.grid_scores_)
plt.show()

new_data = X.ix[:,[1,2,3,4,7,8,9,10,12,13,15,17,18,19,21,26,27,28,29,32,33,34]]
new_data['labels'] = df['label']
new_data.to_csv('selected_features_22.csv', index = None, header=True)


#%% 
#Cross-validated Feature selection using LassoCV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectFromModel
import numpy as np
# load data
df = pd.read_csv('features.csv')
df = df[df.concreteness.notnull()]
X = df.loc[:, df.columns != 'label']
X = X.drop(['sentence_tokenize', 'sentence_lower'], axis=1)
y = df['label']

clf = LassoCV(cv=5)
 
# Set a minimum threshold of 0.70
sfm = SelectFromModel(clf, threshold=0.70)
sfm.fit(X, y)
n_features = sfm.transform(X).shape[1]
 
# Extracting the index of important features
feature_idx = sfm.get_support()

coef_index = sfm.estimator_.coef_
threshold = sfm.threshold_

# %%
new_data = X.ix[:,[29,32]]
new_data['labels'] = df['label']

new_data.to_csv('selected_features_2.csv', index = None, header=True)


# %%
