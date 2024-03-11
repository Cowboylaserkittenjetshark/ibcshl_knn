import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplcatppuccin
import seaborn as sns
from common import DATA_FILE, OUTPUT

data = pd.read_csv(DATA_FILE)

# Drop bad columns
data.drop(
    columns=[
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours",
    ],
    inplace=True,
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
    "OverTime",
]

sns.set_style("whitegrid")
mpl.style.use("mocha")
for cat in cats:
    vals = data[cat]
    labels = vals.unique()
    plt.pie(vals.value_counts(), labels=labels)
    plt.title(cat)
    plt.savefig(
        OUTPUT.joinpath(f"pie/{cat}.svg"), bbox_inches="tight", transparent=False
    )
    plt.savefig(
        OUTPUT.joinpath(f"pie/{cat}.png"), bbox_inches="tight", transparent=False
    )
    plt.clf()
