

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                          --     Imports     --
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
# {{{                              --     Train/Test/Val Splits     --
#···············································································
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# train_ratio = 0.80
# test_ratio = 0.10
# validation_ratio = 0.10

X, y = df.drop(['Y1', 'Y2'], axis=1), df[['Y1', 'Y2']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_ratio/(train_ratio+test_ratio))

# sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
# for train_index, test_index in sss.split(X, y):
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]

# scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────




## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                     --      Ridge Regression     --
#···············································································
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV 
from sklearn.model_selection import RandomizedSearchCV


alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]
regr_cv = RidgeCV(alphas = alpha_ridge)

model_cv = regr_cv.fit(X_valid, y_valid)

model_cv.alpha_

best_model = Ridge(alpha=0.01)

scores = cross_val_score(best_model, X_train, y_train)
model = Ridge()
parameters={'alpha': alpha_ridge}
scores = cross_val_score(best_model, X_test, y_test)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))



Ridge_reg= GridSearchCV(model, parameters, cv=5)

Ridge_reg.fit(X_test, y_test)
Ridge_reg.best_estimator_
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


from scipy.stats import uniform as sp_rand
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {'alpha': sp_rand()}
# create and fit a ridge regression model, testing random alpha values
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100)
rsearch.fit(X_train, y_train)
#
print(rsearch)
# summarize the results of the random parameter search
print(rsearch.best_score_)
print("Alpha: ", rsearch.best_estimator_.alpha)
a = rsearch.best_estimator_.alpha

best_model = Ridge(alpha=a)
best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
print("R2 Training: ", r2_score(y_train, y_train_pred))
print("MSE Training: ", mean_squared_error(y_train, y_train_pred))


y_test_pred = best_model.predict(X_test)
print("R2 Test:", r2_score(y_test, y_test_pred))
print("MSE Test: ", mean_squared_error(y_test, y_test_pred))

 
scores = cross_val_score(best_model, X_train, y_train)
print("Training Scores: ", "%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(best_model, X_test, y_test)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

scores = cross_val_score(rsearch, X_train, y_train)
cv = cross_val_predict(rsearch, X_train, y_train)

mse = mean_squared_error(y_train, cv)
rsquared = r2_score(y_train, cv)

print(mse)
print(rsquared)
print(scores)

## TESTING
rr_pred = cross_val_predict(rsearc






## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     Poly     --
#···············································································
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score 
degrees = [2, 3, 4, 5, 6] # Change degree "hyperparameter" here
normalizes = [True, False] # Change normalize hyperparameter here
best_score = 0
best_degree = 0
for degree in degrees:
    for normalize in normalizes:
        poly_features = PolynomialFeatures(degree = degree)
        X_train_poly = poly_features.fit_transform(X_train)
        polynomial_regressor = LinearRegression(normalize=normalize)
        polynomial_regressor.fit(X_train_poly, y_train)
        scores = cross_val_score(polynomial_regressor, X_train_poly, y_train, cv=5) # Change k-fold cv value here
        if max(scores) > best_score:
            best_score = max(scores)
            best_degree = degree
            best_normalize = normalize


print(best_score)
print(best_normalize)
print(best_degree)



poly_features = PolynomialFeatures(degree = best_degree)
X_train_poly = poly_features.fit_transform(X_train)
best_polynomial_regressor = LinearRegression(normalize=best_normalize)
polynomial_regressor.fit(X_train_poly, y_train)

y_train_preds = polynomial_regressor.predict(X_train_poly)
train_r2 = cross_val_score(polynomial_regressor, X_train, y_train, cv=5)
test_r2 = cross_val_score(polynomial_regressor, X_test, y_test, cv=5)



np.average(test_r2)


#plt.plot(X_test.ravel(), y_test, 'r')
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────
y_test.values


df.head()


X_train.describe()

X_test.describe()
