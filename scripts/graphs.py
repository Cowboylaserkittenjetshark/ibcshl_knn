import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT = ROOT.joinpath("output/")
OUTPUT.mkdir(exist_ok=True)
OUTPUT.joinpath("pie/").mkdir(exist_ok=True)
DATA_FILE = ROOT.joinpath("data/data.csv")

# reading in data
data = pd.read_csv(DATA_FILE)

# drop bad columns
data.drop(
    columns = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
    ], 
    inplace=True
)

# Pie charts
cats = [
    "Attrition",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]
sns.set_style("whitegrid") # Set style for chart
plt.figure(figsize=(10,10)) # Set figure size
for cat in cats:
    vals = data[cat]
    labels = vals.unique()
    plt.pie(vals.value_counts(), labels=labels)
    plt.title(cat)
    plt.savefig(OUTPUT.joinpath(f"pie/{cat}.svg"), bbox_inches="tight", transparent = True)
    plt.savefig(OUTPUT.joinpath(f"pie/{cat}.png"), bbox_inches="tight", transparent = True)
plt.clf()

# Encode
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['BusinessTravel'] = data['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
# one_hot_encoded_data = pd.get_dummies(data, columns = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']) 

data.drop(
    columns = ['Department', 'EducationField', 'JobRole', 'MaritalStatus'], 
    inplace=True
)

svm = sns.heatmap(data.corr())
figure = svm.get_figure()
plt.savefig(OUTPUT.joinpath("heatmap.svg"), bbox_inches="tight", transparent = True)
plt.savefig(OUTPUT.joinpath("heatmap.png"), bbox_inches="tight", transparent = True)
plt.clf()
