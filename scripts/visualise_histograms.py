import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def main():
    df = pd.read_csv("./debug/block_list_histogram.csv")

    column_headers = list(df.columns.values)

    plt.bar(column_headers, df.mean(), yerr=df.std(), capsize=5)
    plt.xlabel("Angle bin / °")
    plt.ylabel("Magnitude")
    plt.show()

def main_2d():
    df_1 = pd.read_csv("./debug/theta_and_psi_histogram.csv")
    df_1= df_1[df_1['Theta'] != 0]


    print(df_1)

    plt.hist2d(df_1["Theta"], df_1["Psi"], bins=(9,9), cmap="gist_heat_r")

    plt.xlabel("Theta / °")
    plt.ylabel("Psi / °")
    plt.colorbar()
    plt.savefig("./debug/theta_psi_hist.png")


if __name__ == "__main__":
    main_2d()