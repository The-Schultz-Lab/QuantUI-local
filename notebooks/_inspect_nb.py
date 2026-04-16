import json
from pathlib import Path


def main():
    notebook_path = Path(__file__).resolve().parent / "molecule_computations.ipynb"
    with notebook_path.open(encoding='utf-8') as f:
        nb = json.load(f)
    print("nbformat: {}.{}".format(nb["nbformat"], nb["nbformat_minor"]))
    print("Total cells: {}".format(len(nb["cells"])))
    for i, cell in enumerate(nb["cells"]):
        cid = cell.get("id", "N/A")
        src_preview = "".join(cell["source"])[:70].replace("\n", " ")
        tags = cell.get("metadata", {}).get("tags", [])
        print("  Cell {}: {} id={}  tags={}  src={}".format(i, cell["cell_type"], cid, tags, repr(src_preview)))


if __name__ == "__main__":
    main()
