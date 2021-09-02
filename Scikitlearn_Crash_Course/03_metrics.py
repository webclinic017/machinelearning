import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# how accurate/ is get all case
from sklearn.metrics import precision_score, recall_score, make_scorer
import warnings
warnings.filterwarnings('ignore')


def min_recall_precision(est, X, y_true, sample_weight=None):
    y_pred = est.predict(X)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)


df = pd.read_csv('data/creditcard.csv')[:80_000]
# print(len(df))  # 284807 origin vs 80000

X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values
# print(f'Shapes of X={X.shape} y={y.shape}, #Fraud cases={y.sum()}')

mod = LogisticRegression(class_weight={0: 1, 1: 2},  max_iter=1000)
pred_sum = mod.fit(X, y).predict(X).sum()
# print(pred_sum)

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v}
                                 for v in np.linspace(1, 20, 30)]},
    scoring={'precision': make_scorer(precision_score),
             'recall_score': make_scorer(recall_score),
             'min_both': min_recall_precision},
    refit='precision',
    return_train_score=True,
    cv=10,   # cross validation
    n_jobs=-1
)

grid_fit = grid.fit(X, y)
# print(recall_score(y, grid_fit.predict(X)))
# print(type(grid.fit(X, y).cv_results_))   # dict
df = pd.DataFrame(grid_fit.cv_results_)
# df.to_csv('data/train_result.csv')
# print(df)

# ------------------- Plotting part -------------------
plt.figure(figsize=(12, 4))
for score in ['mean_test_recall', 'mean_test_precision']:
    plt.plot([_[1] for _ in df['param_class_weight']], df[score], label=score)
plt.legend()
plt.show()
