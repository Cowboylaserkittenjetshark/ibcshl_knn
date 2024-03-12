import pandas as pd
import matplotlib.pyplot as plt
from common import DATA_FILE, OUTPUT, TRANSPARENT

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

for cat in cats:
    vals = data[cat]
    labels = vals.unique()
    plt.pie(vals.value_counts(), labels=labels)
    plt.title(cat)
    plt.savefig(
        OUTPUT.joinpath(f"pie/{cat}.svg"), bbox_inches="tight", transparent=TRANSPARENT
    )
    plt.savefig(
        OUTPUT.joinpath(f"pie/{cat}.png"), bbox_inches="tight", transparent=TRANSPARENT
    )
    plt.clf()
