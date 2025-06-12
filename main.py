import numpy as np
import pandas as pd
import time
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

from source.svm import SVM

path = './datasets/diabetes_dataset.csv'

df = pd.read_csv(path)
y = df['Outcome']
y = y.replace(0, -1)
X = df.drop(['Outcome'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

scaler = StandardScaler()

scaled_x_train = np.array(scaler.fit_transform(X_train))
scaled_x_test = np.array(scaler.transform(X_test))




start = time.perf_counter()
svm = SVM()
svm.fit(scaled_x_train,np.array(y_train) )
svm_pred = svm.predict(np.array(scaled_x_test))

finish = time.perf_counter()
y_test = np.array(y_test)



w = svm.support()

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

# print(f'accuracy:  {accuracy:.2f}')


print(classification_report(y_test, svm_pred))
print('время работы: ' + str(finish-start))




