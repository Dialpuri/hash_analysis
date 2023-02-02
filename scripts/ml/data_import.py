import os
import pandas as pd
import numpy as np


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


def combine_datasets(df_1: pd.DataFrame, df_2: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """
    Expects two dataframes with labels to combine
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


if __name__ == "__main__":
    extract_data_from_folder("./debug/histogram_data_sugars")
