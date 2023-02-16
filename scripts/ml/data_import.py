import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def extract_data_from_folder(base_folder_path: str) -> pd.DataFrame:
    """
    Extract the data from the output of the hash_exec program
    @rtype: pd.DataFrame
    @param base_folder_path: str - expected to include two folders, psi and theta
    """

    theta_folder_path = os.path.join(base_folder_path, "theta")
    psi_folder_path = os.path.join(base_folder_path, "psi")

    combined_df = pd.DataFrame()

    for file in os.scandir(theta_folder_path):
        theta_file_path = os.path.join(theta_folder_path, file.name)
        psi_file_path = os.path.join(psi_folder_path, file.name)

        assert os.path.isfile(theta_file_path) == os.path.isfile(psi_file_path)

        df_theta = pd.read_csv(theta_file_path)
        df_psi = pd.read_csv(psi_file_path)

        df_concat = pd.concat([df_theta, df_psi], axis=1)
        combined_df = pd.concat([combined_df, df_concat], axis=0)

    return combined_df


def combine_datasets_columnwise(df_1: pd.DataFrame, df_2: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """
    Expects two dataframes with labels to combine column_wise
    @param labels: list[str] - must be of length 2
    @param df_1: pd.DataFrame
    @param df_2: pd.DataFrame
    """

    assert len(labels) == 2

    df_1_names = list(df_1)
    df_2_names = list(df_2)

    df_1_renamed = [f"{labels[0]}_{x}" for x in df_1_names]
    df_2_renamed = [f"{labels[1]}_{x}" for x in df_2_names]

    df_1.columns = df_1_renamed
    df_2.columns = df_2_renamed

    return_df = pd.concat([df_1, df_2], axis=1)

    return return_df

def combine_datasets_rowwise_with_type(df_1: pd.DataFrame, df_2: pd.DataFrame, labels: list[str], shuffle: bool) -> pd.DataFrame:
    """
    Expects two dataframes with labels to combine rowwise
    @param labels: list[str] - must be of length 2
    @param df_1: pd.DataFrame
    @param df_2: pd.DataFrame
    """

    assert len(labels) == 2

    additional_column_label = "type"
    df_1[additional_column_label] = np.ones(len(df_1))
    df_2[additional_column_label] = np.zeros(len(df_2))

    return_df = pd.concat([df_1, df_2], axis=0)

    if shuffle:
        return_df = return_df.sample(frac=1).reset_index(drop=True)

    return return_df


def combined_datasets_with_labels(df_1: pd.DataFrame, df_2: pd.DataFrame, shuffle: bool = False) -> pd.DataFrame:
    """
    Combine datasets rowwise with df_1 having a positive label (1) and df_2 having a label of 0.
    @param shuffle: bool
    @param df_1: pd.DataFrame
    @param df_2: pd.DataFrame
    """

    additional_column_label = "type"
    df_1[additional_column_label] = np.ones(len(df_1))
    df_2[additional_column_label] = np.zeros(len(df_2))

    return_df = pd.concat([df_1, df_2], axis=0)

    if shuffle:
        return_df = return_df.sample(frac=1).reset_index(drop=True)

    return return_df


def import_8x8x8_grid(base_folder_path: str, cutoff = None) -> pd.DataFrame:
    grid_size = 8

    output_df_headers = []

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):
            for k in range(0, grid_size, 1):
                header = f"{i}{j}{k}"
                data = header + "_data"
                sugar = header + "_sugar"
                output_df_headers.append(data)
                output_df_headers.append(sugar)

    return_df = pd.DataFrame(columns=output_df_headers)
    return_df.columns = output_df_headers

    counter = 0
    for file in tqdm(os.scandir(base_folder_path), total=len(os.listdir(base_folder_path))):
        if cutoff:
            if counter >= cutoff:
                break
            counter += 1
        tmp_list = []
        headers = ["u", "v", "w", "data", "sugar"]
        df = pd.read_csv(file.path, names=headers)

        for index, row in df.iterrows():
            # new_header = f"{row['u']}{row['v']}{row['w']}"
            tmp_list.append(row['data'])
            tmp_list.append(row['sugar'])

        output_df = pd.DataFrame(tmp_list).T
        output_df.columns = output_df_headers

        return_df = pd.concat([return_df, output_df])

    return return_df


def import_8x8x8_data(df_1_path: str, df_2_path: str, labels: list[str], shuffle: bool) -> pd.DataFrame:
    df_1 = pd.read_csv(df_1_path)
    df_2 = pd.read_csv(df_2_path)

    return combine_datasets_rowwise_with_type(df_1, df_2, labels, shuffle)




def generate_data_and_label_files():
    df = import_8x8x8_data(
        df_1_path="./debug/labelled_points/sugars_data.csv",
        df_2_path="./debug/labelled_points/no_sugars_data.csv",
        labels=["sugars", "nosugars"],
        shuffle=True
    )

    data_df = pd.DataFrame()
    label_df = pd.DataFrame()

    for column in df:
        if "data" in column:
            data_df[column] = df[column]
        elif "sugar" in column:
            label_df[column] = df[column]
        else:
            print(column)

    data_df.to_csv("./debug/labelled_points/data.csv")
    label_df.to_csv("./debug/labelled_points/labels.csv")

def create_data_file():
    df_sugars = import_8x8x8_grid("./debug/labelled_points/sugars")
    df_no_sugars = import_8x8x8_grid("./debug/labelled_points/no_sugars", cutoff=len(df_sugars))

    df_sugars.to_csv("debug/labelled_points/sugars_data.csv")
    df_no_sugars.to_csv("debug/labelled_points/no_sugars_data.csv")

if __name__ == "__main__":
    extract_data_from_folder("./debug/histogram_data_sugars")
