import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import sklearn.metrics as metrics
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from common import DATA_FILE, OUTPUT


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
le = LabelEncoder()
y = le.fit_transform(y)
target_values = le.classes_

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Pipeline applying scaler and knn
clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=15, metric="minkowski", weights="uniform", leaf_size=15, p=1
            ),
        ),
    ]
)


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
'''train_score = {}
test_score = {}
n_neighbors = np.arange(2, 30, 1)
for neighbor in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    train_score[neighbor] = knn.score(X_train, y_train)
    test_score[neighbor] = knn.score(X_test, y_test)
plt.plot(n_neighbors, train_score.values(), label="Train Accuracy")
plt.plot(n_neighbors, test_score.values(), label="Test Accuracy")
plt.xlabel("Number Of Neighbors")
plt.ylabel("Accuracy")
plt.title("KNN: Varying number of Neighbors")
plt.legend()
plt.xlim(0, 33)
plt.ylim(0.60, 0.90)
plt.grid()
plt.savefig(
    OUTPUT.joinpath("neighbors.svg"), bbox_inches="tight", transparent=True
)
plt.savefig(
    OUTPUT.joinpath("neighbors.png"), bbox_inches="tight", transparent=True
)
for key, value in test_score.items():
    if value == max(test_score.values()):
        print(key)'''
k_range = range(1, 40)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k,metric='manhattan')
    knn.fit(X_train,np.ravel(y_train,order='C'))
    y_pred = knn.predict(X_test)
    # appending the accuracy scores in the dictionary named scores.
    scores.append(metrics.accuracy_score(y_test, y_pred))
print(scores)
plt.plot(k_range, scores)
plt.xlabel('Value of K')
plt.ylabel('Testing Accuracy')
plt.savefig(
    OUTPUT.joinpath("minkowski.png"), bbox_inches="tight", transparent=True
)
test_scores = []
train_scores = []
for i in range(1,15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train) 
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))
    
## Training Evaluation
max_train_score = max(train_scores)
# # Store the max train test score index by enumerating through all the scores.
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
# Store the max score in the first curly parenthesis and the indices in the second.
# The lambda function takes the index starting at zero therefore one is added to get the k value.
print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))
## Testing Evaluation
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
## Train Test Evaluation by comparative graph.
plt.figure(figsize=(12,5))
p = sns.lineplot(train_scores,label='Train Score')
p = sns.lineplot(test_scores,label='Test Score')
plt.savefig(
    OUTPUT.joinpath("test.png"), bbox_inches="tight", transparent=True
)
