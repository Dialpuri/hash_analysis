import os
import urllib.error

import wget


def main():
    for entry in os.scandir("./data/RNA_test_structures/PDB Files"):
        file_name = entry.name
        pdb_code = file_name.split(".")[0].replace("pdb", "")

        download_url_base = "https://edmaps.rcsb.org/coefficients/"

        download_url = f"{download_url_base}{pdb_code}.mtz"

        mtz_file_path = os.path.join("data/RNA_test_structures/MTZ Files", f"{pdb_code}_phases.mtz")
        if os.path.isfile(mtz_file_path):
            continue

        try:
            wget.download(download_url, out="data/RNA_test_structures/MTZ Files")
        except urllib.error.HTTPError:
            print(pdb_code, "failed.")

if __name__ == "__main__":
    main()
