import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------- FUNCTIONS ----------------------------------------------------- #
def attr_probs(data, attr):
    attr.append('D')
    df = pd.DataFrame(data=data, columns=attr)
    probs = {
        '0': 0,
        '1': 0
    }
    positive = 0
    negative = 0
    tp = 0

    for index, row in df.iterrows():
        if row[attr[0]] == 1 and row['D'] == 1:
            positive += 1
        elif row[attr[0]] == 0 and row['D'] == 1:
            negative += 1
        tp += 1

    probs['0'] = negative / tp
    probs['1'] = positive / tp

    print(f'Positive: {positive}/{tp}')
    print(f'Negative: {negative}/{tp}')

    return probs


def calc_log2(value):
    if value > 0:
        return np.log2(value)
    return 0


def entropy(list):
    entropy_value = 0

    for value in list:
        entropy_value -= value * calc_log2(value)

    return round(entropy_value, 3)


def unique(df):
    unique_list = np.unique(np.array(df))
    return unique_list


def gain(data, attr):
    column = [attr, 'D']
    df = pd.DataFrame(data=data, columns=column)
    unique_values = unique(df[attr])

    set_positive, set_negative, set_total = 0, 0, 0
    subset_1_positive, subset_1_negative, subset_1_total = 0, 0, 0
    subset_2_positive, subset_2_negative, subset_2_total = 0, 0, 0
    subset_3_positive, subset_3_negative, subset_3_total = 0, 0, 0

    if len(unique_values) == 2:
        for index, row in df.iterrows():
            if row['D'] == 1:
                if row[attr] == unique_values[0]:
                    subset_1_positive += 1
                    subset_1_total += 1
                else:
                    subset_2_positive += 1
                    subset_2_total += 1
                set_positive += 1
            else:
                if row[attr] == unique_values[0]:
                    subset_1_negative += 1
                    subset_1_total += 1
                else:
                    subset_2_negative += 1
                    subset_2_total += 1
                set_negative += 1
            set_total += 1
    else:
        for index, row in df.iterrows():
            if row['D'] == 1:
                if row[attr] == unique_values[0]:
                    subset_1_positive += 1
                    subset_1_total += 1
                elif row[attr] == unique_values[1]:
                    subset_2_positive += 1
                    subset_2_total += 1
                else:
                    subset_3_positive += 1
                    subset_3_total += 1
                set_positive += 1
            else:
                if row[attr] == unique_values[0]:
                    subset_1_negative += 1
                    subset_1_total += 1
                elif row[attr] == unique_values[1]:
                    subset_2_negative += 1
                    subset_2_total += 1
                else:
                    subset_3_negative += 1
                    subset_3_total += 1
                set_negative += 1
            set_total += 1

    entropy_set = entropy([set_positive / set_total, set_negative / set_total])
    entropy_subset_1 = entropy([subset_1_positive / subset_1_total, subset_1_negative / subset_1_total])
    entropy_subset_2 = entropy([subset_2_positive / subset_2_total, subset_2_negative / subset_2_total])

    if attr == 'A':
        print(f'- Entropy Value: {entropy_set} \
                - Positive: {set_positive}/{set_total} \
                - Negative: {set_negative}/{set_total}\n')

    print(f'- Set {attr}:')
    print(f'    - Subset {unique_values[0]}:')
    print(f'        - Positive: {subset_1_positive}/{subset_1_total} \
                    - Negative: {subset_1_negative}/{subset_1_total} \
                    - Entropy: {entropy_subset_1}')

    print(f'    - Subset {unique_values[1]}:')
    print(f'        - Positive: {subset_2_positive}/{subset_2_total} \
                    - Negative: {subset_2_negative}/{subset_2_total} \
                    - Entropy: {entropy_subset_2}')

    gain_value = entropy_set - (subset_1_total / set_total) * entropy_subset_1 - (subset_2_total / set_total) * entropy_subset_2

    if len(unique_values) == 3:
        entropy_subset_3 = entropy([subset_3_positive / subset_3_total, subset_3_negative / subset_3_total])
        print(f'    - Subset {unique_values[2]}:')
        print(f'        - Positive: {subset_3_positive}/{subset_3_total}\
                        - Negative: {subset_3_negative}/{subset_3_total}\
                        - Entropy: {entropy_subset_3}')

        gain_value = entropy_set - (subset_1_total / set_total) * entropy_subset_1 - (subset_2_total / set_total) * entropy_subset_2 - (subset_3_total / set_total) * entropy_subset_3

    print(f'\nSET {attr} GAIN:  {round(gain_value, 3)}\n')
    return round(gain_value, 3)


# ------------------------------------------------------- MAIN ------------------------------------------------------- #

# load data
data = pd.read_csv('depression.csv', sep=',')
# entropy = entropy([4/7, 3/7])
ATTRS = ['A', 'B', 'C']
for attr in ATTRS:
    gain(data, attr)
