import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler



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

lre = LogisticRegression()

#scaler = StandardScaler()
#X_train_standardized = scaler.fit_transform(X_train)


final_w, final_b, all_costs = lre.fit(X_train, y_train)
print(all_costs)

y_pred = lre.predict(X_train)





acc = accuracy(y_train, y_pred)

print(acc)




