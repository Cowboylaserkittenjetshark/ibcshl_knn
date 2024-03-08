import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import DATA_FILE, OUTPUT

# reading in data
data = pd.read_csv(DATA_FILE)

# drop bad columns
data.drop(
    columns = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
        "DailyRate",
        "EnvironmentSatisfaction",
        "JobSatisfaction",
        "MonthlyRate",
        "RelationshipSatisfaction",
        "WorkLifeBalance"
    ], 
    inplace=True
)

# Encode
data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
data['BusinessTravel'] = data['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

one_hot_cats = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']

# plt.figure(figsize=(20,20))
svm = sns.heatmap(data.drop(one_hot_cats, axis=1).corr(), annot=False, annot_kws={"size": 8}) # Heatmap without categories that will be one hot encoded
figure = svm.get_figure()
plt.savefig(OUTPUT.joinpath("heatmap/main.svg"), bbox_inches="tight", transparent = True)
plt.savefig(OUTPUT.joinpath("heatmap/main.png"), bbox_inches="tight", transparent = True)
plt.clf()

# Heatmaps for onehot-ed cols
for cat in one_hot_cats:
    cat_data = data[["Attrition", cat]] # Select only attrition and category to one hot encode
    heatmap_data = pd.get_dummies(cat_data, columns = [cat], prefix='', prefix_sep='') # One hot encode
    svm = sns.heatmap(heatmap_data.corr(), annot=True)
    plt.savefig(OUTPUT.joinpath(f"heatmap/{cat}.svg"), bbox_inches="tight", transparent = True)
    plt.savefig(OUTPUT.joinpath(f"heatmap/{cat}.png"), bbox_inches="tight", transparent = True)
    plt.clf()
