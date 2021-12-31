import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('energy.csv')
df.isnull().sum()

colors = ['#1F2041', '#4B3F72', '#FFC857', '#119DA4', '#19647E']

colors = ["#272932","#4d7ea8","#828489","#9e90a2","#b6c2d9","#efdd8d","#f4fdaf"]
fig, ax = plt.subplots(3,3)



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     Cleaning / Feature Engineering     --
#···············································································
df = df.drop(['Unnamed: 10', 'Unnamed: 11'], axis=1)
# df['Y1R'] = df['Y1'].round().rt_index()
# df['Y2R'] = df['Y2'].round().value_counts().sort_index()
# #                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{              --     Train / Test / Validation Split     --
#···············································································
from sklearn.model_selection import train_test_split

train_ratio = 0.80
test_ratio = 0.10
validation_ratio = 0.10

# X, y = df.drop(['Y1', 'Y2'], axis=1), df[['Y1', 'Y2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                             --   Distributions     --
#···············································································
fig, axes =plt.subplots(3,3, figsize=(15,10))
axes = axes.flatten()
for ax, catplot in zip(axes, df.columns):
    sns.countplot(x=catplot, data=df, ax=ax, palette=colors)

plt.tight_layout()  
plt.show()                     
#}}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                        --      Scatterplot     --
#···············································································
fig, axes =plt.subplots(3,3, figsize=(10,10))
axes = axes.flatten()
for ax, col in zip(axes, df.columns[:-4]):
    sns.scatterplot(x='Y1R', y='Y2R', data=df, ax=ax, hue=col, palette='husl')

plt.tight_layout()  
plt.show() 

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Heatmap     --
#···············································································
fig, ax = plt.subplots(figsize=(10,10))
corr = df.corr()
sns.heatmap(corr, fmt='0.2f',  annot=True)
plt.show()
 
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                            --     PCA     --
#···············································································
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaled_df = X_train
scaled_df[scaled_df.columns] = scaler.fit_transform(scaled_df)

from sklearn.decomposition import PCA
#X_train_PCA = PCA(n_components = 5)

pca = PCA().fit(scaled_df)

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 9, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 10, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()


X_reduced_pca = PCA(n_components=5, random_state=42).fit_transform(scaled_df.values)
#                                                                            }}} 
## ─────────────────────────────────────────────────────────────────────────────

