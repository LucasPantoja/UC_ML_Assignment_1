import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------- Exercise 1 ---------------------------------------------------- #
def unique(df):
    unique_list = np.unique(np.array(df))
    return unique_list


def attr_probs(data, attr):
    probs = {
        '0': 0,
        '1': 0
    }
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

    probs['0'] = subset_1_positive / subset_1_total
    probs['1'] = subset_2_positive / subset_2_total

    return probs, unique_values, set_positive, set_negative, set_total, \
           subset_1_positive, subset_1_negative, subset_1_total, \
           subset_2_positive, subset_2_negative, subset_2_total, \
           subset_3_positive, subset_3_negative, subset_3_total


# ---------------------------------------------------- Exercise 2 ---------------------------------------------------- #
def calc_log2(value):
    if value > 0:
        return np.log2(value)
    return 0


def entropy(list):
    entropy_value = 0

    for value in list:
        entropy_value -= value * calc_log2(value)

    return round(entropy_value, 3)


# ---------------------------------------------------- Exercise 3 ---------------------------------------------------- #
def gain(data, attr):
    probs, unique_values, set_positive, set_negative, set_total, \
    subset_1_positive, subset_1_negative, subset_1_total, \
    subset_2_positive, subset_2_negative, subset_2_total, \
    subset_3_positive, subset_3_negative, subset_3_total = attr_probs(data, attr)

    entropy_set = entropy([set_positive / set_total, set_negative / set_total])
    entropy_subset_1 = entropy([subset_1_positive / subset_1_total, subset_1_negative / subset_1_total])
    entropy_subset_2 = entropy([subset_2_positive / subset_2_total, subset_2_negative / subset_2_total])

    if attr == 'A':
        print(f'- Entropy Value: {entropy_set} \
                - Positive: {set_positive}/{set_total} \
                - Negative: {set_negative}/{set_total}\n')

    print(f'- Set {attr}:')
    print(
        f'    - Subset {unique_values[0]}  "0":({round(subset_1_negative / subset_1_total, 3)}) "1"({round(subset_1_positive / subset_1_total, 3)}):')
    print(f'        - Positive: {subset_1_positive}/{subset_1_total} \
                    - Negative: {subset_1_negative}/{subset_1_total} \
                    - Entropy: {entropy_subset_1}')

    print(
        f'    - Subset {unique_values[1]} "0":({round(subset_2_negative / subset_2_total, 3)}) "1"({round(subset_2_positive / subset_2_total, 3)}):')
    print(f'        - Positive: {subset_2_positive}/{subset_2_total} \
                    - Negative: {subset_2_negative}/{subset_2_total} \
                    - Entropy: {entropy_subset_2}')

    gain_value = entropy_set - (subset_1_total / set_total) * entropy_subset_1 - (
            subset_2_total / set_total) * entropy_subset_2

    if len(unique_values) == 3:
        entropy_subset_3 = entropy([subset_3_positive / subset_3_total, subset_3_negative / subset_3_total])
        print(
            f'    - Subset {unique_values[2]} "0":({round(subset_3_negative / subset_3_total, 3)}) "1"({round(subset_3_positive / subset_3_total, 3)}):')
        print(f'        - Positive: {subset_3_positive}/{subset_3_total}\
                        - Negative: {subset_3_negative}/{subset_3_total}\
                        - Entropy: {entropy_subset_3}')

        gain_value = entropy_set - (subset_1_total / set_total) * entropy_subset_1 - (
                subset_2_total / set_total) * entropy_subset_2 - (subset_3_total / set_total) * entropy_subset_3

    print(f'\nSET {attr} GAIN:  {round(gain_value, 3)}\n')
    return round(gain_value, 3)


# ---------------------------------------------------- Exercise 4 ---------------------------------------------------- #
# load data
csv = pd.read_csv('depression.csv', sep=',')

ATTRS = ['A', 'B', 'C']
for attr in ATTRS:
    gain(csv, attr)


# ---------------------------------------------------- Exercise 5 ---------------------------------------------------- #
data = csv.iloc[:, :-1]
labels = csv.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=7)

decision_tree = DecisionTreeClassifier(criterion='entropy')

decision_tree.fit(X_train, y_train)

rules = export_text(decision_tree, feature_names=ATTRS)
print(rules)


# ---------------------------------------------------- Exercise 6 ---------------------------------------------------- #
# **Answer:**
#
# case: c1 = (A=1, B=0, C=2)<p>
# path: `(C > 1.5) --> (B <= 0.5) --> (A > 0.5) --> class 0`<p>
# _
#
# case: c2 = (A=0, B=0, C=0)<p>
# path: `(C <= 1.5) --> (A <= 0.5) --> (B <= 0.5) --> (C <= 0.5) --> class 1`<p>
# _
#
# case: c3 = (A=0, B=0, C=1)<p>
# path: `(C <= 1.5) --> (A <= 0.5) --> (B <= 0.5) --> (C > 0.5) --> class 1`<p>
# _
#
# case: c4 = (A=1, B=1, C=0)<p>
# path: `(C <= 1.5) --> (A > 0.5) --> (C <= 0.5) --> (B > 0.5) --> class 0`<p>


# ---------------------------------------------------- Exercise 7 ---------------------------------------------------- #
y_pred = decision_tree.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=cm)


TP = cm[1][1]
TN = cm[0][0]
FP = cm[1][0]
FN = cm[0][1]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * ((precision * recall) / (precision + recall))

plot_cm.plot()
plt.savefig('confusion_matrix')
plt.show()
print(f'- True Positive : {round(TP * 100 / len(X_test), 3)}%\
        - True Negative : {round(TN * 100 / len(X_test), 3)}%\
        - False Positive : {round(FP * 100 / len(X_test), 3)}%\
        - False Positive : {round(FN * 100 / len(X_test), 3)}%\
        - Accuracy : {round(accuracy, 3)}\
        - Precision : {round(precision, 3)}\
        - Recall : {round(recall, 3)}\
        - F1 : {round(f1, 3)}')


# ---------------------------------------------------- Exercise 8 ---------------------------------------------------- #
accuracy = []
for random_state in range(100):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=random_state)

    decision_tree = DecisionTreeClassifier(criterion='entropy')

    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))

acc_mean = np.mean(accuracy)
std = np.std(accuracy)
print(f'Accuracy Mean: {round(acc_mean, 3)}')
print(f'Stantard Deviation: {round(std, 3)}')
plt.plot(range(100), accuracy, label='Accuracy')  # Plot some data on the (implicit) axes.
plt.xlabel('Repetition')
plt.ylabel('Accuracy')
plt.title("Accuracy Plot")
plt.legend()
plt.show()
