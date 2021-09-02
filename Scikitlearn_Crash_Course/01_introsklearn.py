from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

# '''
# data / modelling, knn, create object, fit, Pipeline, GridSearch
X, y = load_boston(return_X_y=True)

# model fit
mod = KNeighborsRegressor().fit(X, y)

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('model', KNeighborsRegressor(n_neighbors=1))
])

# pipe.fit(X, y)
# pred = pipe.predict(X)
# plt.scatter(pred, y)
# plt.show()

mod = GridSearchCV(estimator=pipe,
                   param_grid={'model__n_neighbors':
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, cv=3)

# pipeline fit
mod.fit(X, y)
df = pd.DataFrame(mod.cv_results_)
print(df)
# '''
# -----------------------------------------------------------
# print(load_boston()['DESCR'])
