import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Object Creation: Series and DataFrame ################

# Series: One-dimensional ndarray with axis labels (including time series)
s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)

# DataFrame: NumPy array, datetime index, labeled columns
dates = pd.date_range('20210101', periods=6)
# print(dates)  # DatetimeIndex
df = pd.DataFrame(np.random.randn(6, 4), index=dates,
                  columns=list(['Open', 'High', 'Close', 'Low']))
#  ---------- np.random / DataFrame params -----------
# print(df)

df2 = pd.DataFrame({
    "A": 1.0,
    "B": pd.Timestamp('20210102'),
    "C": pd.Series(1, index=list(range(4)), dtype='float32'),
    "D": np.array([3]*4, dtype='int32'),
    "E": pd.Categorical(['test', 'train', 'test', 'train']),
    "F": 'foo'
})
# print(df2)
# print(df2.dtypes)
#  ---------- look like dict / size/ all thing are object -----------

# Viewing data ########################################

# print(df.head(3))
# print(df.tail(3))
# print(df)
# df.set_index('Open', inplace=True)
# print(df.index)
# print(df.columns)
#  ---------- Index([columns]) -----------

#  -- DataFrame: one dtype per column vs numpy: one dtype entire array --
# print(df)
# print(df.to_numpy())

# more expensive if columns has different dtype
# print(df2)
# print(df2.to_numpy())
#  ---------- to_numpy -----------

# print(df.describe())
#  ---------- statistic summary -----------

# print(df.T)
#  ---------- Transpose -----------

# print(df.sort_index(axis=1, ascending=False))
# ----- sort_index / axis=0 mean columns, axis=1 mean row -----

# print(df.sort_values(by='Close', ascending=False))
# ----- sort_values / easy sort columns -----

# #### Selection: manupulate data same as list using [] ####
# : at, iat, loc, iloc :

# ----------------------- Getting ---------------------

# print(df['Open'])
# ------------------------ get 1 column ----------

# slice row
# print(df[1:3])
# ----------- look like slice list

# ----------------------- Selection by label -----------------

# get 1 row
# print(df.loc[dates[0]])
# ----------- make 'dates' example ------

# multi axis by label
# print(df.loc[:, ['Open', 'Close']])

# define exactly dates -> slice
# print(df.loc['2021-01-01': '2021-01-04', ['Open', 'Close']])

# other example
# print(df.loc['2021-01-01', ['Open', 'Close']])

# getting scalar value
# print(df.loc[dates[0], ['Close']])

# fast access
# print(df.at[dates[0], 'Close'])
# ---------------------- Summary: loc, at -- label ----------------

# ----------------------- Selection by position -----------------
# print(df.iloc[3])

# same same list slice
# print(df.iloc[1:5, 0:3])

# NumPy/Python style: 2 list index of row/ column
# print(df.iloc[[1, 2, 4], [0, 1, 2]])
# normalize data -> numpy ????

# only slicing rows
# print(df.iloc[1:3, :])

# only slicing cols
# print(df.iloc[:, 0:3])

# get a value
# print(df.iloc[0, 0])

# fast access
# print(df.iat[1, 1])

# ----------------------- Boolean indexing -----------------
# df label columns:
# print(df[df['Open'] > 0])

# print(df[df > 0])

df3 = df.copy()
df3['Volume'] = [np.random.randint(0, 5) for _ in range(len(df3))]
df['Volume'] = df3['Volume']
# print(df3)

# filtering: isin = is in
# print(df3[df3['Volume'].isin([3])])

# ----------------------- Setting -----------------
# Series : 1 col??? or something

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20210101', periods=6))
# print(s1)

# add new col by Series
df['Change %'] = s1
# print(df)

# # set cell values
# df.at[dates[0], ['High', 'Low']] = 0
# # print(df)

# # notice column content [0:len], trc do phai set index o date
# # or convert date to index format
# df.iat[0, 1] = 0
# print(df)

# # set all columns
# df.loc[:, 'Low'] = np.array([5]*len(df))

df4 = df.copy()
# print(df4)
# # change sign
# print(-df4)

df4[df4 > 0] = -df4
# print(df4)

# ----------------------- Missing data -----------------

# np.nan to represent missing data
# Reindexing a copy of the data.
# change/add/delete the index on a specified axis
df1 = df.reindex(index=dates[0:4], columns=list(df.columns)+["Currency"])
df1.loc[dates[0]:dates[1], 'Currency'] = 1

# # drop
# # df1.dropna(how='any', inplace=True)

# # fill
# df1.fillna(value=5, inplace=True)
# # print(df1)

# print(pd.isna(df1))

# ----------------------- Operations -----------------
# # Stats: Operations in general exclude missing data ------------
# # print(df)

# descriptive statistic
# print(df.mean())    # columns
# print(df.mean(1))   # row

s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
# ----- tan dung ham shift in quant -----
# print(s)
# print(df.sub(s, axis='index'))

# ---- sub??? dataframe -----


# # Applying functions to the data ------------
# print(df)
# print(df.apply(np.cumsum))
# cumsum ??? ----------------

# print(df.apply(lambda x: x.max()-x.min()))

# # Histogramming : value_counts ------------------------
s = pd.Series(np.random.randint(0, 7, size=6))
# print(s)
# print(s.value_counts())

# # String methods ------------
s = pd.Series(["A", "B", "C", "Aaba", "Baca", np.nan, "CABA", "dog", "cat"])
# print(s.str.lower())

# # Merge ------------
df = pd.DataFrame(np.random.randn(10, 4))
# print(df)

# get row
pieces = [df[:3], df[3:7], df[7:]]
# print(pd.concat(pieces))
# print(df[:3])
# print(df[3:7])
# print(df[7:])

# SQL style merges -------------
left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
# print(pd.merge(left, right, on="key"))
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html


# # ----------------------- Grouping -----------------
# Splitting
# Applying
# Combining

# # dictionary
# df = pd.DataFrame({"A": ["foo", "bar", "foo", "bar", "foo",
#                          "bar", "foo", "foo"],
#                   "B": ["one", "one", "two", "three", "two",
#                         "two", "one", "three"],
#                    "C": np.random.randn(8),
#                    "D": np.random.randn(8)})
# print(df.groupby('A').sum())

# # Grouping by multiple columns forms a hierarchical index
# # priority
# print(df.groupby(['A', 'B']).sum())
# print(df.groupby(['B', 'A']).sum())

# # ----------------------- Reshaping -----------------
# Stack: -----------------
# https://stackoverflow.com/questions/29139350/difference-between-ziplist-and-ziplist/29139418
tuples = list(zip(*[["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                    ["one", "two", "one", "two", "one", "two", "one", "two"]]))
# print(len(tuples))
# print(tuples)

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
# print(index)
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
# print(df[:4])

# stack():compresses level, --------------
# stacked DF, Series (có MultiIndex as the index)
stacked = df[:4].stack()
# print(stacked)

# unstack via level -------------------
# default last level
# print(stacked.unstack())

# print(stacked.unstack(1))

# print(stacked.unstack(0))
# -------------------------------------------------
# Pivot tables: -----------------
df = pd.DataFrame({"A": ["one", "one", "two", "three"] * 3,
                   "B": ["A", "B", "C"] * 4,
                   "C": ["foo", "foo", "foo", "bar", "bar", "bar"] * 2,
                   "D": np.random.randn(12),
                   "E": np.random.randn(12)
                   })
# print(df)
df = pd.pivot_table(df, values=['D', 'E'], index=['A', 'B'], columns=['C'])
# print(df)
# df.to_csv('test.csv')

# # ----------------------- Time series -----------------
# DatetimeIndex
rng = pd.date_range('1/1/2021', periods=100, freq='S')
# print(len(rng))

ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)

# # DatetimeIndexResampler
# print(ts.resample('240min'))
# print(ts.resample('240min').sum())

# timezone
rng = pd.date_range('1/1/2021 00:00', periods=5, freq='D')
# print(rng)
ts = pd.Series(np.random.randn(len(rng)), rng)
# print(ts)
ts_utc = ts.tz_localize('UTC')
# print(ts_utc)
# print(ts_utc.tz_convert('US/Eastern'))

rng = pd.date_range('1/1/2021', periods=5, freq='M')
# print(rng)
ts = pd.Series(np.random.randn(len(rng)), index=rng)
# print(ts)
ps = ts.to_period()
# print(ps)
# print(ps.to_timestamp())
# ??? to_period, to_timestamp

# # ----------------------- Categoricals -----------------
df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "raw_grade": [
                  "a", "b", "b", "a", "a", "e"]})
# print(df)
df['grade'] = df['raw_grade'].astype('category')
# rename for meaningful
df["grade"].cat.categories = ["very good", "good", "very bad"]
# print(df['grade'])
# type : category or ...???


df['grade'] = df['grade'].cat.set_categories(
    ['very good', 'good', 'very bad', 'bad', 'medium'])
# print(df['grade'])
# print(df.sort_values(by='grade'))

# # ----------------------- Plotting -----------------
plt.close('all')
ts = pd.Series(np.random.randn(1000),
               index=pd.date_range('1/1/2021', periods=1000))
ts = ts.cumsum()
# ts.plot()
# plt.show()

df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index,
                  columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
# plt.figure()
# df.plot()
# plt.legend(loc='best')
# plt.show()
# figure() vs plot() ???

# # ------------- Getting data in/out --------------
# df.to_csv('foo.csv')
# df = pd.read_csv('foo.csv')
# print(df.tail())

# df.to_excel('foo.xlsx', sheet_name='sheet 1')
# df = pd.read_excel('foo.xlsx', 'sheet 1', index_col=None,
#                    na_values=['NA'], engine='openpyxl')
print(df)
# ---------------------- Summary -----------------------

# Creation: Series, DataFrame
# View data: head, tail, index, columns, to_numpy, describe, sort
# Selection: get, select by label/ position, boolean indexing, set,
# Missing data: reindex, dropna, fillna, isna
# Operations: stats, mean, sub, apply, histogram, string
# Merge: concat, merge
# Grouping: split, apply, combine
# Reshape: stack, pivot table
# Time series: resample, tz_, Series
# Categorials: cat.set_categories, cat.categories
# Plotting: matplotlib
# Getting data in/out: read/ to ...
