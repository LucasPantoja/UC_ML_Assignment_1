import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

csv = pd.read_csv('depression.csv')

data = csv.iloc[:, :-1]
labels = csv.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=7)

classifier = DecisionTreeClassifier(criterion='entropy')

classifier.fit(X_train, y_train)

rules = export_text(classifier, feature_names=['A', 'B', 'C'])
print(rules)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
plot_cm = ConfusionMatrixDisplay(confusion_matrix=cm)


TP = cm[0][0]
FN = cm[0][1]
FP = cm[1][0]
TN = cm[1][1]

plot_cm.plot()
plt.savefig('confusion_matrix')
plt.show()
print(classification_report(y_true=y_test, y_pred=y_pred))
print(f'- True Positive : {TP}\
        - True Negative : {TN}\
        - False Positive : {FP}\
        - False Negative : {FN}')

accuracy = []
for random_state in range(100):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=random_state)

    classifier = DecisionTreeClassifier(criterion='entropy')

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy.append(accuracy_score(y_true=y_test, y_pred=y_pred))


plt.plot(range(100), accuracy, label='Accuracy')  # Plot some data on the (implicit) axes.
plt.xlabel('Repetition')
plt.ylabel('Accuracy')
plt.title("Accuracy Plot")
plt.legend()
plt.show()