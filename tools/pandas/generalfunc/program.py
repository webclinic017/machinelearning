import pandas as pd
import numpy as np


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)


def preparing_data(num):
    df = pd.read_csv('data/Silver_Weekly.csv')
    df.drop('Currency', axis=1, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[-num:]
    return df

# ###########################################################
# ######### Data manipulations ##########
# ###########################################################
# ---------------- pandas.melt -----------------------


def melt():
    tmp = {'A': {0: 'a', 1: 'b', 2: 'c'},
           'B': {0: 1, 1: 3, 2: 5},
           'C': {0: 2, 1: 4, 2: 6}}
    df = pd.DataFrame(tmp)
    print(df)
    # preparing data
    df = preparing_data(5)
    print(df)

    # relocate DataFrame
    df = pd.melt(df, id_vars=['Open', 'Close'], value_vars=[
        'High', 'Low'], ignore_index=False,
        var_name='Cols', value_name='Values')

    # set name, lắp data từ 1 cột...
    df = pd.melt(df, id_vars=['A'], value_vars=['B'],
                 var_name='Hung', value_name='Nguyen')
    print(df.tail())
    print(df)

    # default: variable  value
    df = pd.melt(df, id_vars=['A'], value_vars=['B'])
    print(df)

    # If you have "multi-index columns"
    df.columns = [list('ABC'), list('DEF')]
    print(df)

    df = pd.melt(df, col_level=0, id_vars=['A'], value_vars=['B'])
    print(df)

    df = pd.melt(df, id_vars=[('B', 'E')], value_vars=[('C', 'F')])
    print(df)

# ---------------- pandas.pivot -----------------------


def pivot():
    # Return reshaped DataFrame

    index = 'foo', columns = 'bar', values = 'baz'

    df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                               'two'],
                       'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                       #    'bar': ['A', 'B', 'C', 'D', 'E', 'F'],
                       'baz': [1, 2, 3, 4, 5, 6],
                       'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    df = df.pivot(index='foo', columns='bar', values='baz')
    df = df.pivot(index='foo', columns='bar')['baz']  # same with above
    df = df.pivot(index='foo', columns='bar', values=['baz', 'zoo'])
    print(df)

    # preparing data
    df = preparing_data(3)
    df = df.pivot(index='Open', columns=['High', 'Low'], values='Close')
    print(df)
    '''
      # 1. fill nan value ???
      # 2. using shift value
      # 3. diagonal matrix (features and labels)
      gg: "diagonal matrix features machine learning"
      '''

    df = pd.DataFrame({
        "lev1": [1, 1, 1, 2, 2, 2],
        "lev2": [1, 1, 2, 1, 1, 2],
        "lev3": [1, 2, 1, 2, 1, 2],
        "lev4": [1, 2, 3, 4, 5, 6],
        "values": [0, 1, 2, 3, 4, 5]})
    df = df.pivot(index="lev1", columns=["lev2", "lev3"], values="values")
    df = df.pivot(index=["lev1", "lev2"], columns=["lev3"], values="values")

    df = pd.DataFrame({"foo": ['one', 'one', 'two', 'two'],
                       # Index contains duplicate entries
                       "bar": ['A', 'A', 'B', 'C'],
                       "baz": [1, 2, 3, 4]})
    df = df.pivot(index='foo', columns='bar', values='baz')
    print(df)

# ---------------- pandas.pivot_table -----------------------


def pivot_table():
    # Create a spreadsheet-style pivot table
    # https://www.youtube.com/watch?v=tQRRfIG9UIY
    df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                             "bar", "bar", "bar", "bar"],
                      "B": ["one", "one", "one", "two", "two",
                            "one", "one", "two", "two"],
                       "C": ["small", "large", "large", "small",
                             "small", "large", "small", "small",
                             "large"],
                       "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                       "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})
    table = pd.pivot_table(df, values='D', index=['A', 'B'],
                           # Sum of array elements over a given axis.
                           columns=['C'], aggfunc=np.sum)  # , fill_value=0.0

    # The next example aggregates by taking the mean across multiple columns
    table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
                           aggfunc={'D': np.mean,
                                    'E': np.mean})
    print(table)

    # calculate multiple types of aggregations for any given value column
    table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
                           aggfunc={'D': np.mean,
                                    'E': [min, max, np.mean]})
    print(table)

    '''
      # 1. label the data
      # 2. ~ reshape data by pivot table
      # 3. using aggfunc
      # 4. fill nan | zero | inf values
      '''
    # preparing data
    # index='foo', columns='bar', values='baz'
    # Index contains duplicate entries will be removed
    # auto sort values in columns 'High'

    df = preparing_data(5)
    print(df)
    table = pd.pivot_table(df, index='Open', columns=['High', 'Low'],
                           values=['Close'], aggfunc=np.sum,
                           fill_value=0.0)

    table = pd.pivot_table(df, index=['High', 'Low'],
                           values=['Open', 'Close', 'Volume'],
                           aggfunc={'Open': np.mean, 'Close': np.mean,
                           'Volume': np.mean})
    print(table)
# ---------------- pandas.crosstab -----------------------
# ---------------- pandas.cut -----------------------
# ---------------- pandas.qcut -----------------------


# ---------------- pandas.merge -----------------------
# ---------------- pandas.merge_ordered -----------------------


# ---------------- pandas.merge_asof -----------------------

# ---------------- pandas.concat -----------------------

# ---------------- pandas.get_dummies -----------------------
# ---------------- pandas.factorize -----------------------
# ---------------- pandas.unique -----------------------
# ---------------- pandas.wide_to_long -----------------------

# ###########################################################
# ######### Top-level missing data ##########
# ###########################################################


# ###########################################################
# ######### Top-level conversions ##########
# ###########################################################

# ###########################################################
# ######### Top-level dealing with datetimelike ##########
# ###########################################################

# ###########################################################
# ######### Top-level dealing with intervals ##########
# ###########################################################

# ###########################################################
# ######### Top-level evaluation ##########
# ###########################################################

# ###########################################################
# ######### Hashing ##########
# ###########################################################

# ###########################################################
# ######### Testing ##########
# ###########################################################
