import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import  tqdm



def main():
    df = pd.read_csv("./debug/block_list_histogram.csv")

    column_headers = list(df.columns.values)

    plt.bar(column_headers, df.mean(), yerr=df.std(), capsize=5)
    plt.xlabel("Angle bin / °")
    plt.ylabel("Magnitude")
    plt.show()

def main_2d():

    for csv in tqdm(os.scandir("./debug/histogram_data/2d"), total=len(os.listdir("./debug/histogram_data/2d"))):
        df_1 = pd.read_csv(csv.path)
        df_1 = df_1[df_1['Theta'] != 0]

        plt.hist2d(df_1["Theta"], df_1["Psi"], bins=(9,9), cmap="gist_heat_r")

        plt.xlabel("Theta / °")
        plt.ylabel("Psi / °")
        plt.colorbar()
        # plt.savefig("./debug/theta_psi_hist.png")
        plt.gca().set_aspect('equal')
        plt.savefig(os.path.join("./debug/histogram_data/2d_images", csv.name.replace(".csv",".png")), dpi = 200)
        plt.cla()
        plt.clf()

if __name__ == "__main__":
    main_2d()