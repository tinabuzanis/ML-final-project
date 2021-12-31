
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Imports / Reading Data     --
#···············································································
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD


original_df = pd.read_csv('creditcard.csv')
original_df.head()

X_original = original_df.drop('Class', axis=1)
y_original = original_df['Class']

print('No Frauds', round(original_df['Class'].value_counts()[0]/len(original_df) * 100,2), '% of the dataset')
print('Frauds', round(original_df['Class'].value_counts()[1]/len(original_df)*100,2), '% of the datatset')
 #                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 --     Balancing classes for EDA     --
#···············································································
b_df = original_df.sample(frac=1) 
fraud_df = b_df.loc[b_df['Class'] == 1]
non_fraud_df = b_df.loc[b_df['Class'] == 0][:492]

b_df = pd.concat([fraud_df, non_fraud_df])
b_df = b_df.sample(frac = 1)

## -- comparison of correlation matrices-- 
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 20))

corr = original_df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix", fontsize=20)

balanced_corr = b_df.corr()
sns.heatmap(balanced_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('Balanced Correlation Matrix', fontsize=20)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{            --     Most +/- Correlated Features     --
#···············································································
colors = ["#0101DF", "#DF0101"]

negative_corr= ['V10', 'V12', 'V14', 'V17']
positive_corr = ['V2', 'V4', 'V11', 'V19']

fraud = b_df.loc[b_df['Class'] == 1]
not_fraud = b_df.loc[b_df['Class'] == 0]

j = [0, 0, 1, 1]
k = [2, 3, 2, 3]
kde_idx = [1, 2, 5, 6]
box_idx = [3, 4, 7, 8]

## - negative correlations
sns.set_style('whitegrid')
plt.figure()

fig, ax = plt.subplots(ncols=4, nrows=2,  figsize=(20, 15))
i = 0


fig.suptitle('Most Negatively Correlated Features', fontsize=20, weight='bold')
#fig.subplots_adjust(wspace=0.5)
#fig.subplots_adjust(hspace=0.5)
for col in negative_corr:
    #for idx in kde_idx:
    plt.subplot(2,4,kde_idx[i])
    sns.kdeplot(fraud[col], bw_method=0.5, label='Fraud', color='r')
    sns.kdeplot(not_fraud[col], bw_method=0.5, label='Not Fraud', color='b')
    sns.boxplot(x='Class', y=col, data=b_df, ax=ax[j[i],k[i]], palette=colors)
    plt.xlabel(col, fontsize=15)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend()
    
    #plt.subplot(2,4,2)
    i+=1

## - positive correlations 

sns.set_style('whitegrid')
plt.figure()

fig, ax = plt.subplots(ncols=4, nrows=2,  figsize=(20, 15))
i = 0


fig.suptitle('Most Positively Correlated Features', fontsize=20, weight='bold')
#fig.subplots_adjust(wspace=0.5)
#fig.subplots_adjust(hspace=0.5)
for col in positive_corr:
    #for idx in kde_idx:
    plt.subplot(2,4,kde_idx[i])
    sns.kdeplot(fraud[col], bw_method=0.5, label='Fraud', color='r')
    sns.kdeplot(not_fraud[col], bw_method=0.5, label='Not Fraud', color='b')
    sns.boxplot(x='Class', y=col, data=b_df, ax=ax[j[i],k[i]], palette=colors)
    plt.xlabel(col, fontsize=15)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend()
    
    #plt.subplot(2,4,2)
    i+=1
 

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --     Removing Outliers     --
#···············································································
# Removing outliers from Fraud & Not Fraud with IQR * 1.5 as a guide

cols = b_df.columns[:-1]
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

remove_non_fraud_outliers = remove_outliers(b_df, 0, True)
remove_all_outliers = remove_outliers(remove_non_fraud_outliers, 1, True)

remove_non_fraud_outliers = remove_outliers(remove_all_outliers, 0, True)
remove_all_outliers = remove_outliers(remove_non_fraud_outliers, 1, True)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                    --     Scaling Removed Outliers DF    --
#···············································································
from sklearn.preprocessing import RobustScaler

rm = remove_all_outliers
rscaler = RobustScaler()

rm['Amount'] = rscaler.fit_transform(rm['Amount'].values.reshape(-1,1))
rm['Time'] = rscaler.fit_transform(rm['Time'].values.reshape(-1,1))

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
#  {{{                  --     PCA, TSNE, TruncatedSVD     --
#···············································································
X1 =  rm.drop('Class', axis = 1)
y1 = rm['Class']

# t0 = time.time()
# X_tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto').fit_transform(X1.values)
# t1 = time.time()
# print("TSNE took {:.2} s".format(t1 - t0))

t0 = time.time()
X_pca = PCA(n_components=2).fit_transform(X1.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

t0 = time.time()
X_svd = TruncatedSVD(n_components=2, algorithm='randomized').fit_transform(X1.values)
t1 = time.time()
print('Truncated SVD took {:.2} s'.format(t1 - t0))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --     PCA, TSNE, TruncatedSVD Plots     --
#···············································································
# TSNE doesn't play nice with qtconsole

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
# ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y1 == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=(y1 == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax1.set_title('t-SNE', fontsize=14)

# ax1.grid(True)

# ax1.legend(handles=[blue_patch, red_patch])

# PCA scatter plot
ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y1 == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_pca[:,0], X_pca[:,1], c=(y1 == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y1 == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_svd[:,0], X_svd[:,1], c=(y1 == 1), cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.show()

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


