import pandas as pd
import diffmod as dm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

fn = 'CVD_balanced.csv'

df = pd.read_csv(fn)

print(df.shape)

X = df.drop(columns=['TenYearCHD', 'exng', 'caa', 'ldl_cholestrol', 'hdl_cholestrol', 'Triglyceride', 'CPK_MB_Percentage'])  # Features
#X = df.drop(columns=['TenYearCHD'])  # Features
y = df['TenYearCHD']  # Target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
'''

print(f'Y_train: {np.shape(y_train)}')
print(f'Y_test: {np.shape(y_test)}')

#pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
#pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print()
dm.rf(X_train, X_test, y_train, y_test)
print()
dm.lda(X_train, X_test, y_train, y_test)
print()
dm.gda(X_train, X_test, y_train, y_test)
print()
dm.knn(X_train, X_test, y_train, y_test)
print()
dm.svm(X_train, X_test, y_train, y_test)
print()
dm.lsvm(X_train, X_test, y_train, y_test)
print()
dm.dt(X_train, X_test, y_train, y_test)
print()
dm.gnb(X_train, X_test, y_train, y_test)
print()
dm.lgr(X_train, X_test, y_train, y_test)
#print()
#dm.lnr(X_train, X_test, y_train, y_test)
print()
dm.gbc(X_train, X_test, y_train, y_test)
print()
dm.xgb(X_train, X_test, y_train, y_test)
print()
dm.lgbm(X_train, X_test, y_train, y_test)
print()
dm.cbt(X_train, X_test, y_train, y_test)
print()
