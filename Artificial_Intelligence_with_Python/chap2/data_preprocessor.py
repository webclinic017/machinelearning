import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    Binarizer, scale,
    MinMaxScaler, normalize
)


def preprocessor(use_fake=True, periods=100):
    if use_fake:
        input_data = np.array([[5.1, -2.9, 3.3],
                               [-1.2, 7.8, -6.1],
                               [3.9, 0.4, 2.1],
                               [7.3, -9.9, -4.5]])

        # sampl = np.random.uniform(low=0.5, high=13.3, size=(50,))
        # input_data = np.random.rand(4, 3)

        # print(input_data)
    else:
        df = pd.read_csv('data\\GBPUSD_Daily.csv', index_col=0,
                         parse_dates=True).tail(periods)
        # print(df.tail())
        # print(df.index)
        # print(df['Close'].mean())

        # print(type(df['Close']))  # pandas.core.series.Series
        # input_data = df['Close'].to_numpy()

        input_data = df.tail().to_numpy()   # just tail

        # print(input_data)

    # -------------------------------------------------------------------------
    # Binarization: not need fit
    data_binarized = Binarizer(threshold=2.1).transform(input_data)

    # print("data_binarized:\n", data_binarized)

    # data_binarized = Binarizer(threshold=2.1).fit_transform(input_data)
    # print("data_binarized:\n", data_binarized)

    # -------------------------------------------------------------------------
    # Mean removal
    chosen_axis = 1

    # print("Mean and std BEFORE")
    # print(input_data.mean(axis=chosen_axis), input_data.std(axis=chosen_axis), sep=" | ")

    # print("Mean and std AFTER")
    data_scaled = scale(input_data)
    # print(data_scaled.mean(axis=chosen_axis), data_scaled.std(axis=chosen_axis), sep=" | ")

    # -------------------------------------------------------------------------
    # Scaling: need fit
    data_scaler_minmax = MinMaxScaler(feature_range=(0, 1))

    # transformer = data_scaler_minmax.fit(input_data)
    # data_scaled_minmax = transformer.transform(input_data)
    # print(data_scaled_minmax)

    data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
    # print(data_scaled_minmax)

    # -------------------------------------------------------------------------
    # Normalize data

    # Least Absolute Deviations: sum of absolute values is 1 in each row, resistant to outliers
    data_normalized_l1 = normalize(input_data, norm='l1')
    # print("\nL1 normalized data:\n", data_normalized_l1)

    # print(np.sum(data_normalized_l1, axis=1))   # represents columns

    # least squares: sum of squares is 1.
    # data_normalized_l2 = normalize(input_data, norm='l2')
    # print("\nL2 normalized data:\n", data_normalized_l2)

    data_normalized_l2 = normalize(input_data)
    # print("\nL2 normalized data:\n", data_normalized_l2)


def main():
    preprocessor(False)


if __name__ == "__main__":
    main()
