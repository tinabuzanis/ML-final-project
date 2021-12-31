
## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                         --     Imports      --
#···············································································
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('energy.csv')

df = df.drop(['Unnamed: 10', 'Unnamed: 11'], axis=1)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                             --     Train/Test/Val Splits     --
#···············································································
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# train_ratio = 0.80
# test_ratio = 0.10
# validation_ratio = 0.10

X, y = df.drop(['Y1', 'Y2'], axis=1), df[['Y1', 'Y2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio))

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)

#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

# ============================================================================#

#                               PLS REGRESSION

# ============================================================================#

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                      --      PLS Regression     --
#···············································································
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

plsr = PLSRegression(n_components=8, scale=False)
plsr.fit(X_train, y_train)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                 -- Tuning Components Manually    --
#···············································································
def optimize_pls(X, y, n_comp):
     pls = PLSRegression(n_components=n_comp, scale=False)
     cv = cross_val_predict(pls, X, y)
     rsquared = r2_score(y, cv)
     mse = mean_squared_error(y, cv)
     rpd = y.std()/np.sqrt(mse)

     return (cv, rsquared, mse, rpd)
 
# test with up to 8 components

rsquares = []
mses = []
rpds = []
xticks = np.arange(1, 9)
for n_comp in xticks:
    cv, rsquared, mse, rpd = optimize_pls(X_train, y_train, n_comp)
    rsquares.append(rsquared)
    mses.append(mse)
    rpds.append(rpd)


# Plot the mses
def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()

plot_metrics(mses, 'MSE', 'min')
plot_metrics(rsquares, 'R-Squared', 'max')

rpds


y_cv, r2, mse, rpd = optimize_pls(X_train, y_train, 7)

print('R2: %0.4f, MSE: %0.4f, RPD: %0.4f' %(r2, mse, rpd))
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────



## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                              --          --
#···············································································
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

best_r2 = 0
best_ncmop = 0
for n_comp in range(1, 9):
  my_plsr = PLSRegression(n_components=n_comp, scale=False)
  my_plsr.fit(X_train,y_train)
  preds = my_plsr.predict(X_test)

  r2 = r2_score(preds, y_test)
  if r2 > best_r2:
    best_r2 = r2
    best_ncomp = n_comp

print(best_r2, best_ncomp)


best_model = PLSRegression(n_components=7, scale=False)
best_model.fit(X_train, y_train)

train_preds = best_model.predict(X_train)
print(cross_val_score(best_model, y_train, train_preds))
test_preds = best_model.predict(X_test)
print(cross_val_score(best_model, y_test, test_preds))


#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────

from sklearn.metrics import precision_recall_curve

pls_pred = cross_val_predict(best_model, X_train, y_train)
#precision, recall, threshold = precision_recall_curve(y_train[:,0], pls_pred)
# Moving on to something elseo
best_model.score(X_train, y_train)
best_model.score(X_test, y_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(best_model, X_test, y_test)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


