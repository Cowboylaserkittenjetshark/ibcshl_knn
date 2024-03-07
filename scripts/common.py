import pathlib

ROOT = pathlib.Path(__file__).parent.parent.resolve()
OUTPUT = ROOT.joinpath("output/")
OUTPUT.mkdir(exist_ok=True)
OUTPUT.joinpath("pie/").mkdir(exist_ok=True)
OUTPUT.joinpath("heatmap/").mkdir(exist_ok=True)
DATA_FILE = ROOT.joinpath("data/data.csv")
