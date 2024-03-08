import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)
import matplotlib.pyplot as plt
import numpy as np

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
train_score = {}
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
        print(key)
