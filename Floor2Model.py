import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("WiFiLocator_Floor2.csv")
x_train = data[['AP1', 'AP2', 'AP3', 'AP4', 'AP5', 'AP6', 'AP7', 'AP8', 'AP9', 'AP10', 'AP11', 'AP12', 'AP13', 'AP14',
               'AP15', 'AP16', 'AP17', 'AP18', 'AP19', 'AP20', 'AP21', 'AP22', 'AP23', 'AP24', 'AP25', 'AP26', 'AP27', 'AP28', 'AP29', 'AP30',
               'AP31', 'AP32', 'AP33']]
y_train = data[['Room']]

rf_clf = RandomForestClassifier(n_estimators=40, random_state=10)
rf_clf.fit(x_train, y_train)

joblib.dump(rf_clf, 'floor2Model.pkl')
'''2층 모델 파일'''
