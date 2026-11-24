"""Download EchoNet-LVH from Redivis with the Redivis Python client.

Run from the project root (the client isn't a project dependency):

    uv run --no-project --with redivis python scripts/download_echonet_lvh.py <owner.dataset>           # list tables
    uv run --no-project --with redivis python scripts/download_echonet_lvh.py <owner.dataset> <table>   # download one

<owner.dataset> is the dataset's fully qualified Redivis reference, e.g. "someowner.echonet_lvh". The dataset page's
URL id (cchq-0srz1fy9a) is not accepted. Find the reference on the dataset page: Export > Programmatic shows a
snippet containing it, and the page header shows the owner and dataset name.

The first run prints a sign-in link; log in with the Redivis account that was approved for access.
Files already in the output folder are skipped, so an interrupted download can be re-run.
"""

import sys
from pathlib import Path

import redivis

OUT_DIR = Path("data/echonet-lvh")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)

    redivis.authenticate()
    dataset = redivis.dataset(sys.argv[1])

    if len(sys.argv) < 3:
        print("Tables in the dataset (re-run with one of these names to download its files):")
        for table in dataset.list_tables():
            print(f"  {table.name}")
        return

    table = dataset.table(sys.argv[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading files of '{table.name}' to {OUT_DIR.resolve()}")
    table.download_files(OUT_DIR, overwrite=False, progress=True)


if __name__ == "__main__":
    main()
