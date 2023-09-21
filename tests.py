import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('healthcare.csv')


from decisiontree import ID3
from logisticregression import LogisticRegression


df = pd.read_csv('healthcare.csv')
X = df.drop(columns=['Id', 'Outcome']).to_numpy()
y = df['Outcome'].to_numpy()



def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy





X_train, X_test, y_train, y_test = train_test_split(X,y)




clf = LogisticRegression()


final_w, final_b = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)




acc = accuracy(y_test, y_pred)

print(acc)




