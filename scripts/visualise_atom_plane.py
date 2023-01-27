import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    file_name = "./debug/atom_positions.csv"

    df = pd.read_csv(file_name)

    # plot raw data
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(df.x, df.y, df.z, color='b')

    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)

    x, y = np.meshgrid(x, y)

    z = 2.57817 * x + 1.62003 * y + -84.8838

    ax.plot_surface(x, y, z)

    plt.show()

if __name__ == "__main__":
    main()