import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("WiFiLocator_Floor4.csv")

Xdata = data[['AP1', 'AP2', 'AP3', 'AP4', 'AP5', 'AP6', 'AP7', 'AP8', 'AP9', 'AP10', 'AP11', 'AP12', 'AP13', 'AP14',
               'AP15', 'AP16', 'AP17', 'AP18', 'AP19', 'AP20', 'AP21', 'AP22', 'AP23', 'AP24', 'AP25', 'AP26', 'AP27', 'AP28', 'AP29', 'AP30',
               'AP31', 'AP32', 'AP33', 'AP34', 'AP35', 'AP36', 'AP37', 'AP38', 'AP39', 'AP40', 'AP41', 'AP42', 'AP43', 'AP44']]
Ydata = data[['Room']]

x_train, x_valid, y_train, y_valid = train_test_split(Xdata, Ydata, test_size=0.2, shuffle=True, stratify=Ydata, random_state=34)

rf_clf = RandomForestClassifier(n_estimators=40, random_state=5)
rf_clf.fit(x_train, y_train)


pred = rf_clf.predict(x_valid)
print(pred)
print(y_valid)
print(accuracy_score(y_valid, pred))
'''4층 모델 파일'''
