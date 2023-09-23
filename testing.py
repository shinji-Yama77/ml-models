from baggingclassify import BaggingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score





df = pd.read_csv('healthcare.csv')

X = df.drop(columns=['Id', 'Outcome']).to_numpy()
y = df['Outcome'].to_numpy()




X_train, X_test, y_train, y_test = train_test_split(X,y)


clf = BaggingClassifier(num_trees=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)






