import os
import urllib.error
from tqdm import tqdm
import wget


def main():
    for entry in tqdm(os.scandir("./data/DNA_test_structures/PDB_Files"), total=len(os.listdir("./data/DNA_test_structures/PDB_Files"))):
        file_name = entry.name
        pdb_code = file_name.split(".")[0].replace("pdb", "")

        download_url_base = "https://edmaps.rcsb.org/coefficients/"

        download_url = f"{download_url_base}{pdb_code.lower()}.mtz"

        mtz_file_path = os.path.join("data/DNA_test_structures/MTZ_Files", f"{pdb_code}_phases.mtz")
        if os.path.isfile(mtz_file_path):
            continue

        try:
            wget.download(download_url, out="data/DNA_test_structures/MTZ_Files")
        except urllib.error.HTTPError as e :
            print(pdb_code, "failed.", e)

if __name__ == "__main__":
    main()
