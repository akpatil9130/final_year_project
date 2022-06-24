import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib
'''Importing the Dataset'''
data = pd.read_csv('data.csv')

'''Taking Care of Missing Values'''
data.isnull().sum()

'''Taking Care of Duplicate Values'''
data_dup = data.duplicated().any()

data = data.drop_duplicates()
data_dup = data.duplicated().any()


cate_val = []
cont_val = []
for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)


data['cp'].unique()


cate_val.remove('sex')
cate_val.remove('target')
data = pd.get_dummies(data,columns = cate_val,drop_first=True)

data.head()

st = StandardScaler()
data[cont_val] = st.fit_transform(data[cont_val])

X = data.drop('target',axis=1)

y = data['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,
                                               random_state=42)
'''Logistic Regression'''

log = LogisticRegression()
log.fit(X_train,y_train)

y_pred1 = log.predict(X_test)

accuracy_score(y_test,y_pred1)

svm = svm.SVC()

svm.fit(X_train,y_train)

y_pred2 = svm.predict(X_test)

accuracy_score(y_test,y_pred2)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

y_pred3=knn.predict(X_test)

accuracy_score(y_test,y_pred3)


score = []

for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test,y_pred))




plt.plot(score)
plt.xlabel("K Value")
plt.ylabel("Acc")
plt.show()
data = pd.read_csv('data.csv')
data = data.drop_duplicates()
X = data.drop('target',axis=1)
y=data['target']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,
                                                random_state=42)


''' Decision Tree Classifier'''
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred4= dt.predict(X_test)
accuracy_score(y_test,y_pred4)

''' Random Forest Classifier'''
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred5= rf.predict(X_test)
accuracy_score(y_test,y_pred5)
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred6 = gbc.predict(X_test)
accuracy_score(y_test,y_pred6)
final_data = pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                          'ACC':[accuracy_score(y_test,y_pred1)*100,
                                accuracy_score(y_test,y_pred2)*100,
                                accuracy_score(y_test,y_pred3)*100,
                                accuracy_score(y_test,y_pred4)*100,
                                accuracy_score(y_test,y_pred5)*100,
                                accuracy_score(y_test,y_pred6)*100]})

print(final_data)
sns.barplot(final_data['Models'],final_data['ACC'])
X=data.drop('target',axis=1)
y=data['target']
rf = RandomForestClassifier()
rf.fit(X,y)
new_data = pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
     'slope':2,
    'ca':2,
    'thal':3,
},index=[0])
print(new_data)
p = rf.predict(new_data)
if p[0]==0:
    print("No Disease")
else:
    print("Disease")


joblib.dump(rf, 'model_joblib_heart')
model = joblib.load('model_joblib_heart')
model.predict(new_data)
data.tail()