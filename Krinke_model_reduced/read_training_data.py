import numpy as np
import pandas as pd


def read_training_data(filename, train_to_predict):
    """
    Reads the training data from the output.xlsx file.
    :param filename: string, the location of the file containing the training data
    :param train_to_predict: string, which z data to read from the training set
    :return:
    """
    # read training data from the excel file into a pd dataframe
    df = pd.read_excel(filename)

    # get a grid of the x and y values from the training data
    x_idx    = np.linspace(1, 13, 13)
    x_val    = list(zip(list(df["α [°]"]), list(df["β [°]"])))
    x_val    = [x for i, x in enumerate(x_val) if x not in x_val[:i]]
    y        = list(df["vf [mm/min]"].unique())

    # loop over the grid of x and y values and find the corresponding z values in the training data
    z_ratio, z_singleheight, z_doubleheight = [], [], []
    for [alpha, beta] in x_val:
        for speed in y:
            selected_row = df[(df['α [°]'] == alpha) & (df['β [°]'] == beta) & (df['vf [mm/min]'] == speed)]
            z_ratio.append(list(selected_row['Verhältnis [-]'])[0])
            z_singleheight.append(list(selected_row['höhe einf. [mm]'])[0])
            z_doubleheight.append(list(selected_row['höhe dopp. [mm]'])[0])

    # based on the train_to_predict parameter, determine which which z data to read from the training set
    if train_to_predict == "k_ratios":
        Z = np.array(z_ratio).reshape(13, 8).T
    elif train_to_predict == "single_heights":
        Z = np.array(z_singleheight).reshape(13, 8).T
    elif train_to_predict == "double_heights":
        Z = np.array(z_doubleheight).reshape(13, 8).T
    else:
        raise ValueError("Invalid value for train_to_predict")

    return x_idx, y, Z
