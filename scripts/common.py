import pathlib
import matplotlib as mpl
import mplcatppuccin
import seaborn as sns

ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT = ROOT.joinpath("output/")
OUTPUT.mkdir(exist_ok=True)
OUTPUT.joinpath("pie/").mkdir(exist_ok=True)
OUTPUT.joinpath("heatmap/").mkdir(exist_ok=True)
DATA_FILE = ROOT.joinpath("data/data.csv")

# Plot style
TRANSPARENT = False
sns.set_style("whitegrid")
mpl.style.use("mocha")
