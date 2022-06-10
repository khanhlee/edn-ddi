import os
import pandas as pd
import numpy as np
data_dir = 'DDI'

df = pd.read_csv(os.path.join(data_dir, 'Neuron_input.csv'))
df.head()

X = df.drop(['Drug1_ID', 'Drug1', 'Drug2_ID', 'Drug2', 'Y'], axis=1)
y = df.Y

y = y.replace(86,0)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True)

conf_matrices = np.zeros((86, 86))
acc_cv_scores = []
auc_cv_scores = []
for train, test in kfold.split(X, y):
    svm_model = RandomForestClassifier()  
    ## evaluate the model
    svm_model.fit(X.iloc[train], y[train])
    # evaluate the model
    true_labels = np.asarray(y[test])
    predictions = svm_model.predict(X.iloc[test])
    conf_matrices = [[conf_matrices[i][j] + confusion_matrix(true_labels, predictions)[i][j]
               for j in range(len(conf_matrices[0]))] for i in range(len(conf_matrices))]
    acc_cv_scores.append(accuracy_score(true_labels, predictions))
    auc_cv_scores.append(roc_auc_score(true_labels, predictions, average='macro', multi_class='ovr'))
    
print('Accuracy = ', np.mean(acc_cv_scores))
print('AUC = ', np.mean(auc_cv_scores))