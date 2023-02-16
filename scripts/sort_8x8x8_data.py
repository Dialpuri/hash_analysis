import os
import pandas as pd
import shutil
from multiprocessing import Pool
from tqdm import tqdm


def get_file_list(path: str) -> list[str]:
    return [x.path for x in os.scandir(path)]


def worker(file_path: str):
    column_names = ["u", "v", "w", "value", "sugar"]

    try:
        df = pd.read_csv(file_path, header=None, sep=",", names=column_names)
        # df.Name = df.sugar.replace(r'\s+', ' ', regex=True)

        # df.columns = column_names
    except ValueError as e:
        print(f"Error with {file_path} - {e}")
        return
    except IsADirectoryError as e:
        return

    sugar_point_only = df[df["sugar"] != "X"]

    split_path = file_path.split("/")
    file_name = split_path[-1]
    file_path = '/'.join(split_path[:-1])

    folder_name = "sugars" if len(sugar_point_only) != 0 else "nosugars"
    folder_path = os.path.join(file_path, folder_name)

    if os.path.isfile(file_path+"/"+file_name):
        return

    try:
        shutil.move(file_path+"/"+file_name, folder_path)
    except IsADirectoryError:
        print(file_path+"/"+file_name, "->", folder_path)
        return

def main():
    file_list = get_file_list("./debug/32x32x32_points")

    # worker("./debug/labelled_points/3GVN_9993.csv")
    # worker("./debug/labelled_points/3GVN_93.csv")

#
    with Pool() as pool:
        x = list(tqdm(pool.imap_unordered(worker, file_list), total=len(file_list)))


if __name__ == "__main__":
    main()
