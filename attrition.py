import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder


ROOT = pathlib.Path(__file__).parent.resolve().joinpath("data")

# reading in data
data = pd.read_csv(ROOT.joinpath("data.csv"))
feature_names = [
"Age",
"BusinessTravel",
"Department",
"DistanceFromHome",
"Education",
"EducationField",
"EnvironmentSatisfaction",
"Gender",
"HourlyRate",
"JobInvolvement",
"JobLevel",
"JobRole",
"JobSatisfaction",
"MaritalStatus",
"MonthlyIncome",
"NumCompaniesWorked",
"Over18",
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
            'Department',
            'EducationField',
            'Gender',
            'JobRole',
            'MaritalStatus',
            'Over18',
            'OverTime',
        ]
    ),
    (OrdinalEncoder(),
        [
            'BusinessTravel'
        ]
    ),
    remainder='passthrough'
)

# Pipeline applying scaler and knn
clf = Pipeline(
    steps=[("column_transformer", column_transformer), ("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=11, metric="Minkowski"))]
)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
