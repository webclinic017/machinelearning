import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = pd.Series([1, 2, 3, 4, np.nan, 6, 7])
# print(s)

df = pd.DataFrame([[1, 2, 3, 4], [np.nan, 6, 7, 8]])
df.columns = ['col1', 'col2', 'col3', 'col4']
# print(df)

df = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])
df.columns = ['col1', 'col2']
# print(df)

# collect date range (more interesting way)
dates = pd.date_range('20210101', periods=5)
# print(dates)

# view randn random (random normal)
df = pd.DataFrame(np.random.randn(5, 5), index=dates, columns=list('ABCDE'))
# print(df)

df = pd.DataFrame(np.random.randn(5, 4), index=dates,
                  columns=['Open', 'High', 'Low', 'Close'])
# print(df)


# create dict from needed data
df2 = pd.DataFrame(
    {
        # "A": 1.0,
        "A": range(1, 5),
        # "B": pd.Timestamp("20130102"),
        "B": pd.date_range('20210101', periods=4),
        "C": pd.Series(range(4, 0, -1), index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
# print(df2)

# getting a cross section using a label
# print(df.loc[dates[0:3]])

# print(df.loc[dates[0:3], ['High', 'Low']])

# print(df.loc[dates[0], ['High', 'Low']])

# print(df.loc[dates[0], 'High'])
# print(df.at[dates[0], 'High'])

# print(df)
# print(df.iloc[3])
# print(df.iloc[1:3, 0:3])

# print(df.iloc[[1, 3, 4], [0, 3]])

# print(df.iloc[:, 0:3])
# print(df.iloc[:, [0, 2]])

# print(df.iloc[1:3, :])
# print(df.iloc[[1, 3], :])

# print(df)
# print(df.iloc[0, 0])
# print(df.iat[0, 0])

# print(df[df > 0])
df2 = df.copy()
df2['Volume'] = [1000, 200, 400, 100, 300]

# filter zero way
# print(df2[df2['Volume'].isin([200])])

s1 = pd.Series([1, 2, 3, 4, 5], index=pd.date_range('20210101', periods=5))
# print(s1)

df2['pct'] = s1
# print(df2)

df2.at[dates[0], "pct"] = 0

df2.iat[1, 1] = 0

df2.loc[:, 'Volume'] = np.array([500]*len(df2))

df2[df2 > 0] = -df2

print(df2)
