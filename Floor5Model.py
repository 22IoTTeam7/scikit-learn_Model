import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

Xtrain = pd.read_csv()
Ytrain = pd.read_csv()

rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(Xtrain, Ytrain)

print(rf_clf.predict())

'''5층 모델 파일'''