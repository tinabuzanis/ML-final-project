# import EDA
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

from undersample import original_Xtrain, original_ytrain, original_Xtest, original_ytest

df = pd.read_csv('creditcard.csv')

rscaler = RobustScaler()
df['Amount'] = rscaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = rscaler.fit_transform(df['Time'].values.reshape(-1,1))

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Data Prep     --
#···············································································



#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     Outlier Removal (?)     --{{{
#···············································································
cols = df.columns[:-1]
fraud_outliers = []
#fraud_index_list = []
#i_ = []
#total_count = 0

def remove_outliers(v, fraud, remove):
    vcopy = v
    total_count = 0
    for col in cols:
       # print('===========================================================')
       # print('                             {}                            '.format(col))
       # print('===========================================================')
        vf = v[v['Class'] == fraud]
        v_out = v[col].loc[v['Class'] == fraud].values
        q25, q75 = np.percentile(v_out, 25), np.percentile(v_out, 75)
        v_IQR = q75 - q25
        #print('Q25: {} | Q75: {}'.format(q25, q75))
        #print('IQR: {}'.format(v_IQR))

        
        
        v_cutoff = v_IQR * 1.5
        v_lower, v_upper = q25 - v_cutoff, q75 + v_cutoff
        #print('Cut Off: {}'.format(v_cutoff))
        #print('Lower: {}'.format(v_lower))
        #print('Upper: {}'.format(v_upper))
        
        out = [x for x in v_out if x < v_lower or x > v_upper]
        fraud_outliers.append(out)
        #print('Outliers: {}'.format(out))
       # print('Num of Outliers: {}'.format(len(out)))
        total_count += len(out)
        
        if (remove==True):
            vcopy = v.drop(v[(v[col] > v_upper) | (v[col] < v_lower)].index)
    
    print("TOTAL OUTLIERS: ", total_count)
        
    return vcopy

remove_non_fraud_outliers = remove_outliers(df, 0, True)
remove_all_outliers = remove_outliers(remove_non_fraud_outliers, 1, True)

remove_non_fraud_outliers = remove_outliers(remove_all_outliers, 0, True)
remove_all_outliers = remove_outliers(remove_non_fraud_outliers, 1, True)


#           
#                                                                            }}}}}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     Making new ODF     --
#···············································································
original_df = remove_all_outliers

X_original = original_df.drop('Class', axis=1)
y_original = original_df['Class']
sss = StratifiedKFold(n_splits=5)

for train_index, test_index in sss.split(X_original, y_original):
    print("Train: ", train_index, "Test: ", test_index)
    original_Xtrain, original_Xtest = X_original.iloc[train_index], X_original.iloc[test_index]
    original_ytrain, original_ytest = y_original.iloc[train_index], y_original.iloc[test_index]

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     SMOTE     --
#···············································································
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from undersample import log_reg_params


print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))

# List to append the score and then find the average
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

log_reg_sm = LogisticRegression()




rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)


# Implementing SMOTE Technique 
# Cross Validating the right way
# Parameters
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
for train, test in sss.split(original_Xtrain, original_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(original_Xtrain.iloc[train], original_ytrain.iloc[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(original_Xtrain.iloc[test])
    
    accuracy_lst.append(pipeline.score(original_Xtrain.iloc[test], original_ytrain.iloc[test]))
    precision_lst.append(precision_score(original_ytrain.iloc[test], prediction))
    recall_lst.append(recall_score(original_ytrain.iloc[test], prediction))
    f1_lst.append(f1_score(original_ytrain.iloc[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain.iloc[test], prediction))
    
print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(accuracy_lst)))
print("precision: {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 45)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


labels = ['No Fraud', 'Fraud']
smote_prediction = best_est.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))



## ───────────────────────────────────── ▼ ─────────────────────────────────────
 # {{{                          --    Test  Results     --
#···············································································
from undersample import knears_neighbors, svc, tree_clf, X_test, y_test, log_reg, grid_log_reg
from sklearn.metrics import confusion_matrix

sm = SMOTE(sampling_strategy='minority', random_state=42)


Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)

# Logistic Regression
t0 = time.time()
log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm.fit(Xsm_train, ysm_train)
t1 = time.time()
print("Fitting oversample data took :{} sec".format(t1 - t0))


# Logistic Regression fitted using SMOTE technique
y_pred_log_reg = log_reg_sm.predict(X_test)

# Other models fitted with UnderSampling
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)


log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
svc_cf = confusion_matrix(y_test, y_pred_svc)
tree_cf = confusion_matrix(y_test, y_pred_tree)

fig, ax = plt.subplots(2, 2,figsize=(22,12))


sns.heatmap(log_reg_cf, ax=ax[0][0], annot=True, fmt='g',cmap=plt.cm.Blues)
ax[0, 0].set_title("Logistic Regression \n Confusion Matrix", fontsize=14)
ax[0, 0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0, 0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(kneighbors_cf, ax=ax[0][1], annot=True, fmt='g', cmap=plt.cm.Blues)
ax[0][1].set_title("KNearsNeighbors \n Confusion Matrix", fontsize=14)
ax[0][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[0][1].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(svc_cf, ax=ax[1][0], annot=True, fmt='g', cmap=plt.cm.Blues)
ax[1][0].set_title("Suppor Vector Classifier \n Confusion Matrix", fontsize=14)
ax[1][0].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][0].set_yticklabels(['', ''], fontsize=14, rotation=360)

sns.heatmap(tree_cf, ax=ax[1][1], annot=True, fmt='g', cmap=plt.cm.Blues)
ax[1][1].set_title("DecisionTree Classifier \n Confusion Matrix", fontsize=14)
ax[1][1].set_xticklabels(['', ''], fontsize=14, rotation=90)
ax[1][1].set_yticklabels(['', ''], fontsize=14, rotation=360)


plt.show()



from sklearn.metrics import accuracy_score

y_pred = log_reg.predict(X_test)
undersample_score = accuracy_score(y_test, y_pred)



y_pred_sm = best_est.predict(original_Xtest)
oversample_score = accuracy_score(original_ytest, y_pred_sm)


d = {'Technique': ['Random UnderSampling', 'Oversampling (SMOTE)'], 'Score': [undersample_score, oversample_score]}
final_df = pd.DataFrame(data=d)

score = final_df['Score']
final_df.drop('Score', axis=1, inplace=True)
final_df.insert(1, 'Score', score)

# Note how high is accuracy score it can be misleading! 
final_df
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Results     --
#···············································································
from sklearn.metrics import classification_report


print('Logistic Regression:')
print(classification_report(y_test, y_pred_log_reg))

print('KNears Neighbors:')
print(classification_report(y_test, y_pred_knear))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_svc))

print('Support Vector Classifier:')
print(classification_report(y_test, y_pred_tree))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

