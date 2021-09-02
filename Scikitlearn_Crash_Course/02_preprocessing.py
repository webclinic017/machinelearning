from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# '''
# df = pd.read_csv('data/drawndata1.csv')
# X = df[['x', 'y']].values
# y = df['z'] == "a"
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# print(df.tail())
# print(X[-5:])

# # ------------- After scaling -------------
# # X_new = StandardScaler().fit_transform(X)
# X_new = QuantileTransformer(n_quantiles=100).fit_transform(X)
# plt.scatter(X_new[:, 0], X_new[:, 1], c=y)
# plt.show()
# # ------------- End After scaling -------------

# # ------------- Probability Distribution -------------
# x = np.random.exponential(10, 1000)+np.random.normal(0, 1, 1000)
# plt.hist((x-np.mean(x))/np.std(x), 30)
# plt.show()


def plot_output(scaler):
    # pipeline creator
    pipe = Pipeline([
        ('scale', scaler),
        ('model', KNeighborsClassifier(n_neighbors=20, weights='distance'))
    ])
    # prediction
    pred = pipe.fit(X, y).predict(X)
    # plotting
    plt.figure(figsize=(9, 3))
    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Original Data")

    plt.subplot(132)
    X_tfm = scaler.transform(X)
    plt.scatter(X_tfm[:, 0], X_tfm[:, 1], c=y)
    plt.title("Transformed Data")

    plt.subplot(133)
    X_new = np.concatenate([
        np.random.uniform(0, X[:, 0].max(), (5000, 1)),
        np.random.uniform(0, X[:, 1].max(), (5000, 1))
    ], axis=1)
    y_proba = pipe.predict_proba(X_new)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=y_proba[:, 1], alpha=0.7)
    plt.title("Predicted Data")
    plt.show()
    pass


# fit_transform vs transform
# https://www.google.com/search?q=fit_transform+vs+transform&oq=fit_transform+vs+t&aqs=chrome.0.0j69i57j0l2j0i22i30l2j0i390.1181j0j9&sourceid=chrome&ie=UTF-8
plot_output(StandardScaler())
plot_output(QuantileTransformer(n_quantiles=100))
# '''

# --------- a Classification task not linear -----------
df = pd.read_csv('data/drawndata2.csv')
# '''
X = df[['x', 'y']].values
y = df['z'] == "a"
# plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()
pipe = Pipeline([
    # ('scale', QuantileTransformer(n_quantiles=100)),

    # sometime using X^2 feature instead of
    ('scale', PolynomialFeatures()),
    ('model', LogisticRegression())
])

pred = pipe.fit(X, y).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=pred)
plt.show()
# '''
# --------- a simple example -----------
# reshape(-1, 1) ~ reshape(4, 1)
arr = np.array(['low', 'low', 'high', 'medium']).reshape(-1, 1)
enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
enc.fit_transform(arr)
enc.transform([['zero']])
# print()

# -----------
# df = pd.read_clipboard()  # copy csv
