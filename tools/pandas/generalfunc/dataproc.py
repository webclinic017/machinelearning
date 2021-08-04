import pandas as pd


tmp = {'A': {0: 'a', 1: 'b', 2: 'c'},
       'B': {0: 1, 1: 3, 2: 5},
       'C': {0: 2, 1: 4, 2: 6}}
df = pd.DataFrame(tmp)
# print(df)

# ---------------- pandas.melt -----------------------
# df = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'], ignore_index=False)
# print(df)

# # set name, lắp data từ 1 cột...
# df = pd.melt(df, id_vars=['A'], value_vars=['B'],
#              var_name='Hung', value_name='Nguyen')

# # default: variable  value
# df = pd.melt(df, id_vars=['A'], value_vars=['B'])
# print(df)

# df.columns = [list('ABC'), list('DEF')]
# print(df)

# df = pd.melt(df, col_level=0, id_vars=['A'], value_vars=['B'])
# print(df)

# df = pd.melt(df, id_vars=[('B', 'E')], value_vars=[('C', 'F')])
# print(df)
# ---------------- pandas.pivot -----------------------
