import pandas as pd

# Read the coefficients from the included csv files
COEFFICIENTS_K = pd.read_csv("krinke_model_coefficients/k_ratio_new.csv", index_col=0, skipinitialspace=True)
COEFFICIENTS_H = pd.read_csv("krinke_model_coefficients/single_layer_height.csv", index_col=0, skipinitialspace=True)


def s(coefficients, rowname, colname):
    """
    Function to search for a specific coefficient in a pandas dataframe.
    :param coefficients: pd.Dataframe, containing the coefficients to search over
    :param rowname: string, the name of the column
    :param colname: string, the name of the column
    :return: float, value found at row x col in coefficients
    """
    value = coefficients.loc[rowname][colname]
    value = float(value.replace(',', '.'))
    return value


def evaluate(coefficients, alpha, beta, vf):
    """
    Function to evaluate the Krinke equation. Depending on the coefficients you input, you can calculate, for example,
    the height of a single layer or coefficient used when determining the layer height for multiple layers as discussed
    in the paper.
    :param coefficients: pd.Dataframe, containing the coefficients to search over
    :param alpha: float, alpha angle, see paper for explanation
    :param beta: float, beta angle
    :param vf: float, the welding speed in mm/min
    :return: float, the output value of the Krinke equation based on the coefficients
    """
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


def calculate_multi_layer_height(number_of_layers, alpha, beta, vf):
    """
    Function to calculate the height of multiple layers by combining the model outputs of single layer heights with the
    coefficients.
    :param number_of_layers: int, the number of layers
    :param alpha: float, alpha angle, see paper for explanation
    :param beta: float, beta angle
    :param vf: float, the welding speed in mm/min
    :return: float, the estimated height of the specified number of layers
    """
    k_ratios = evaluate(COEFFICIENTS_K, alpha, beta, vf)
    h_single = evaluate(COEFFICIENTS_H, alpha, beta, vf)

    if number_of_layers == 0:
        return k_ratios
    else:
        return h_single + ((number_of_layers-1) * h_single * k_ratios)


if __name__ == "__main__":
    layers, alpha, beta, speed = 3, 90, 55, 300
    result = calculate_multi_layer_height(layers, alpha, beta, speed)
    print("When welding at {}mm/min in the position alpha={}°, beta={}°, the estimated height of {} layer(s) is {}mm."
          .format(speed, alpha, beta, layers, result)
          )
