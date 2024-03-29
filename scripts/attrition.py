import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common import DATA_FILE, OUTPUT, TRANSPARENT
from mplcatppuccin.palette import load_color

color = load_color("mocha", "overlay1")

# Reading in data
data = pd.read_csv(DATA_FILE)

# Encode
data["OverTime"] = data["OverTime"].map({"Yes": 1, "No": 0})
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data["BusinessTravel"] = data["BusinessTravel"].map(
    {"Non-Travel": 0, "Travel_Rarely": 1, "Travel_Frequently": 2}
)
data["MaritalStatus"] = data["MaritalStatus"].map(
    {"Married": 1, "Single": 0, "Divorced": 0}
)
feature_names = [
    "Age",
    "BusinessTravel",
    "DistanceFromHome",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "MaritalStatus",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "StockOptionLevel",
    "TotalWorkingYears",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

X = data[feature_names]
y = data["Attrition"]
print(X.isna().sum())
le = LabelEncoder()
y = le.fit_transform(y)
target_values = le.classes_
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Pipeline applying scaler and knn
'''clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=19, metric="manhattan", weights="uniform", leaf_size=15
            ),
        ),
    ]
)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))'''
classifier = KNeighborsClassifier(n_neighbors=15, metric = "manhattan")
classifier.fit(X_train, np.ravel(y_train,order='C'))
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy =  metrics.accuracy_score(y_test,y_pred)*100
print(accuracy)
for dMetric in ["minkowski", "euclidean", "manhattan"]:
    plt.grid(c=color)
    k_range = range(1, 40)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k,metric=dMetric)
        knn.fit(X_train,np.ravel(y_train,order='C'))
        y_pred = knn.predict(X_test)
        # appending the accuracy scores in the dictionary named scores.
        scores.append(metrics.accuracy_score(y_test, y_pred))
    print(scores)
    plt.plot(k_range, scores)
    plt.xlabel('Value of K')
    plt.ylabel('Testing Accuracy')
    plt.savefig(
        OUTPUT.joinpath(f"{dMetric}.png"), bbox_inches="tight", transparent=TRANSPARENT
    )
    plt.clf()

test_scores = []
train_scores = []
for i in range(1,40):
    knn = KNeighborsClassifier(i, metric='manhattan')
    knn.fit(X_train,y_train) 
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
## Train Test Evaluation by comparative graph.
plt.figure(figsize=(12,5))
p = sns.lineplot(train_scores,label='Train Score')
p = sns.lineplot(test_scores,label='Test Score')
plt.savefig(
    OUTPUT.joinpath("test.png"), bbox_inches="tight", transparent=TRANSPARENT
)
error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i, metric="manhattan")
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    # Create a plot of Mean error versus kvalue.
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.savefig(
    OUTPUT.joinpath("error.png"), bbox_inches="tight", transparent=TRANSPARENT
)
knn_cv = KNeighborsClassifier(n_neighbors=15)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
#train model with cv of 10
cv_scores = cross_val_score(knn_cv, X, np.ravel(y,order='C'), cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
print(np.mean(cv_scores))
scaler = StandardScaler()
scaler.fit(X)
# Train with 10 fold cross validation by an outer k value ranges and nested cross validation scores.
X = scaler.transform(X)
scores = []
k_range = range(1, 40)
for k in k_range:
#train model with cv of 10
    knn_cv = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn_cv, X, np.ravel(y,order='C'), cv=10)
#print each cv score (accuracy) and average them
    print(k)
    print(cv_scores)
    print(np.mean(cv_scores))
