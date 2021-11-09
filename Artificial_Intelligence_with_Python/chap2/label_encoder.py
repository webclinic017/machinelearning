import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def label_encoding(use_fake=True, test_labels=['green', 'red', 'black'], periods=100):
    if use_fake:
        # Sample input labels
        input_labels = ['red', 'black', 'red',
                        'green', 'black', 'yellow', 'white']
        print(input_labels)
    else:
        df = pd.read_csv('data\\GBPUSD_Daily.csv', index_col=0,
                         parse_dates=True).tail(periods)
        df['Change'] = df['Close'].pct_change()

        # label for df Change
        input_labels = ['strong bull', 'bull',
                        'neutral', 'bear', 'strong bear']

        # filter Change values: T.B.D

        # print(df.tail())
        # print(df.index)
        # print(df['Close'].mean())

        # print(type(df['Close']))  # pandas.core.series.Series
        # input_data = df['Close'].to_numpy()

        # input_data = df.tail().to_numpy()   # just tail

        # print(input_data)

    # Create label encoder and fit the labels
    encoder = LabelEncoder()
    encoder.fit(input_labels)

    # Print the mapping
    print("\nLabel mapping:")
    for i, item in enumerate(encoder.classes_):
        print(item, '-->', i)

    # Encode a set of labels using the encoder
    test_labels = test_labels
    encoded_values = encoder.transform(test_labels)
    print("\nLabels =", test_labels)
    print("Encoded values =", list(encoded_values))

    # Decode a set of values using the encoder
    encoded_values = [3, 0, 4, 1]   # fucking shit need tobe handle
    decoded_list = encoder.inverse_transform(encoded_values)
    print("\nEncoded values =", encoded_values)
    print("Decoded labels =", list(decoded_list))


def main():
    # label_encoding(True)
    label_encoding(False, test_labels = ['strong bull', 'bull', 'neutral'])


if __name__ == "__main__":
    main()
