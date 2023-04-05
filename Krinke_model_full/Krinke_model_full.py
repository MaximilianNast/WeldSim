import pandas as pd
import csv

# Read the coefficients from the included csv files
COEFFICIENTS_K = pd.read_csv("krinke_model_coefficients/k_ratio_new.csv", index_col=0, skipinitialspace=True)
COEFFICIENTS_H = pd.read_csv("krinke_model_coefficients/single_layer_height.csv", index_col=0, skipinitialspace=True)


# Function to search for a specific coefficient
def s(coefficients, rowname, colname):
    """
    :param coefficients: pd.Dataframe, containing the coefficients to search over
    :param rowname: string, the name of the column
    :param colname: string, the name of the column
    :return: float, value found at row x col in coefficients
    """
    value = coefficients.loc[rowname][colname]
    value = float(value.replace(',', '.'))
    return value


# Function to calculate a single output of the model
def calculate_model_output(coefficients, alpha, beta, vf):
    c = coefficients

    output = (
                     (s(c, "j", "w_a") * alpha ** 2 + s(c, "k", "w_a") * alpha + s(c, "l", "w_a")) * vf ** 3 +
                     (s(c, "j", "x_a") * alpha ** 2 + s(c, "k", "x_a") * alpha + s(c, "l", "x_a")) * vf ** 2 +
                     (s(c, "j", "y_a") * alpha ** 2 + s(c, "k", "y_a") * alpha + s(c, "l", "y_a")) * vf +
                     (s(c, "j", "z_a") * alpha ** 2 + s(c, "k", "z_a") * alpha + s(c, "l", "z_a"))
             ) * beta + \
             (
                     (s(c, "j", "w_b") * alpha ** 2 + s(c, "k", "w_b") * alpha + s(c, "l", "w_b")) * vf ** 3 +
                     (s(c, "j", "x_b") * alpha ** 2 + s(c, "k", "x_b") * alpha + s(c, "l", "x_b")) * vf ** 2 +
                     (s(c, "j", "y_b") * alpha ** 2 + s(c, "k", "y_b") * alpha + s(c, "l", "y_b")) * vf +
                     (s(c, "j", "z_b") * alpha ** 2 + s(c, "k", "z_b") * alpha + s(c, "l", "z_b"))
             )

    return output


def calculate_layer_height(number_of_layers, alpha, beta, vf):
    k_ratios = calculate_model_output(COEFFICIENTS_K, alpha, beta, vf)
    h_single = calculate_model_output(COEFFICIENTS_H, alpha, beta, vf)

    if number_of_layers == 0:
        return k_ratios
    else:
        return h_single + ((number_of_layers-1) * h_single * k_ratios)


if __name__ == "__main__":
    print(calculate_layer_height(1, 90, 55, 300))
