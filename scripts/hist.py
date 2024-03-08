import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common import DATA_FILE, OUTPUT

OUTPUT.joinpath("hist/").mkdir(exist_ok=True)
data = pd.read_csv(DATA_FILE)

# drop bad columns
data.drop(
    columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
    ],
    inplace=True,
)

# Vis for onehot-ed cols
cats = ["Department", "EducationField", "JobRole", "MaritalStatus", "OverTime"]
for cat in cats:
    cat_data = data[["Attrition", cat]]

    sns.histplot(data=data, x="Attrition", hue=cat, kde=True)
    plt.savefig(
        OUTPUT.joinpath(f"hist/{cat}.svg"), bbox_inches="tight", transparent=True
    )
    plt.savefig(
        OUTPUT.joinpath(f"hist/{cat}.png"), bbox_inches="tight", transparent=True
    )
    plt.clf()
