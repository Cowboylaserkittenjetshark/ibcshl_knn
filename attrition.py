import pandas as pd
import pathlib
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
ROOT = pathlib.Path(__file__).parent.resolve().joinpath("data")

# reading in data
data = pd.read_csv(ROOT.joinpath("data.csv"))
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
feature_names = [
"Age",
#"BusinessTravel",
#"Department",
"DistanceFromHome",
"Education",
#"EducationField",
"EnvironmentSatisfaction",
"Gender",
"HourlyRate",
"JobInvolvement",
"JobLevel",
#"JobRole",
"JobSatisfaction",
#"MaritalStatus",
"MonthlyIncome",
"NumCompaniesWorked",
#"Over18",
"OverTime",
"PercentSalaryHike",
"PerformanceRating",
"RelationshipSatisfaction",
"StockOptionLevel",
"TotalWorkingYears",
"WorkLifeBalance",
"YearsAtCompany",
"YearsInCurrentRole",
"YearsSinceLastPromotion",
"YearsWithCurrManager"
]

X = data[feature_names]
y = data["Attrition"]
le = LabelEncoder()
y = le.fit_transform(y)
target_values = le.classes_
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Column transformer
column_transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'),
        [
            #'Department',
            #'EducationField',
            'Gender',
            #'JobRole',
            #'MaritalStatus',
            #'Over18',
            'OverTime',
        ]
    ),
    (OrdinalEncoder(),
        [
            #'BusinessTravel'
        ]
    ),
    remainder='passthrough'
)

# Pipeline applying scaler and knn
clf = Pipeline(
    steps=[("column_transformer", column_transformer), ("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=15, metric="manhattan"))]
)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
'''train_score = {}
test_score = {}
n_neighbors = np.arange(2, 30, 1)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_score[neighbor]=knn.score(X_train, y_train)
    test_score[neighbor]=knn.score(X_test, y_test)
plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
plt.xlabel("Number Of Neighbors")
plt.ylabel("Accuracy")
plt.title("KNN: Varying number of Neighbors")
plt.legend()
plt.xlim(0, 33)
plt.ylim(0.60, 0.90)
plt.grid()
plt.show()
for key, value in test_score.items():
    if value==max(test_score.values()):
        print(key)'''
knn_pipe = Pipeline([('mms', MinMaxScaler()),
                     ('knn', KNeighborsClassifier())])
params = [{'knn__n_neighbors': [9, 11, 13, 15],
         'knn__weights': ['uniform', 'distance'],
         'knn__leaf_size': [15, 20]}]
gs_knn = GridSearchCV(knn_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5)
gs_knn.fit(X_train, y_train)
gs_knn.best_params_
# find best model score
print(gs_knn.score(X_train, y_train))