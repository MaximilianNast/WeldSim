import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import string
from read_training_data import*


def train(training_data_file, train_to_predict, omit_training_data_indices=[]):
    """
    Function that trains the model on the training data. You can omit training data, i.e. for validation.
    :param training_data_file:
    :param train_to_predict:
    :param omit_training_data_indices:
    :return:
    """

    x, y, Z = read_training_data(training_data_file, train_to_predict)
    X, Y = np.meshgrid(x, y)

    x_flat, y_flat, z_flat = X.flatten(), Y.flatten(), Z.flatten()

    xyz_points = np.column_stack((x_flat, y_flat, z_flat))

    # convert to a list of lists
    xyz_list = xyz_points.tolist()

    xyz_list_no_nan = np.array([point for point in xyz_list if not any(np.isnan(point))])
    x = xyz_list_no_nan[:, 0]
    y = xyz_list_no_nan[:, 1]
    z = xyz_list_no_nan[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    x = normalize_data(x, x_min, x_max)
    y = normalize_data(y, y_min, y_max)
    z = normalize_data(z, z_min, z_max)

    # remove elements
    x = [x[i] for i in range(len(x)) if i not in omit_training_data_indices]
    y = [y[i] for i in range(len(y)) if i not in omit_training_data_indices]
    z = [z[i] for i in range(len(z)) if i not in omit_training_data_indices]

    xpopt, xpcov = scipy.optimize.curve_fit(function, (x, y), z)
    xperr = np.sqrt(np.diag(xpcov))

    dfc = pd.DataFrame(xpopt, columns=["popt"], index=[char for char in string.ascii_lowercase][0:10])
    dfc.to_csv("krinke_model_reduced_coefficients/"+train_to_predict+"_model_coefficients.csv")

    dfm = pd.DataFrame([[x_min, x_max], [y_min, y_max], [z_min, z_max]], columns=["min", "max"], index=["x", "y", "z"])
    dfm.to_csv("krinke_model_reduced_coefficients/"+train_to_predict+"_xyz_min_max.csv")

    return xpopt, dfm, (x, y, z)


def function(X, a, b, c, d, e, f, g, h, i, j):
    """ The function that gets trained to fit the data. """
    x, y = X
    return a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2 + g * x ** 3 + h * x ** 2 * y + i * x * y ** 2 + j * y ** 3


def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def denormalize_data(norm_data, min_val, max_val):
    return norm_data * (max_val - min_val) + min_val


def normalize_alpha_beta(alpha, beta, min_val, max_val):
    ab_comb = [[90, 55, 1], [90, 45, 2], [90, 30, 3], [90, 45,]]


def evaluate(alpha_beta_combination, alpha, beta, speed, coefficient_file, min_max_file):
    """
    Evaluates a single model output for a given combination of alpha, beta, and welding speed based on the input
    coefficient file and min_max file.
    :param alpha_beta_combination: list, the combination of alpha, beta
    :param speed: float, the welding speed in mm/min
    :param coefficient_file: string, the location of the file containing the model coefficients
    :param min_max_file: string, the location of the file containing the model min and max values for normalization
    :return: float, the model output
    """
    # read the trained model values
    coefficients = pd.read_csv(coefficient_file).iloc[:, 1]
    x_min, y_min, z_min = pd.read_csv(min_max_file).iloc[:, 1]
    x_max, y_max, z_max = pd.read_csv(min_max_file).iloc[:, 2]

    # normalize x,y input data
    x = normalize_data(alpha_beta_combination, x_min, x_max)
    print(x)
    print(normalize_alpha_beta(alpha, beta, x_min, x_max))
    y = normalize_data(speed, y_min, y_max)

    # evaluate model output
    output = function((x, y), *coefficients)

    # denormalize model output z data
    output_norm = denormalize_data(output, z_min, z_max)

    return output_norm


def calculate_multi_layer_height(number_of_layers, alpha_beta_combination, alpha, beta, vf):
    """
    Function to calculate the height of multiple layers by combining the model outputs of single layer heights with the
    coefficients.
    :param number_of_layers: int, the number of layers
    :param alpha_beta_combination:
    :param alpha: float, alpha angle, see paper for explanation
    :param beta: float, beta angle
    :param vf: float, the welding speed in mm/min
    :return: float, the estimated height of the specified number of layers
    """

    c_dir = "krinke_model_reduced_coefficients/"
    h_single = evaluate(alpha_beta_combination = alpha_beta_combination,
                        alpha = alpha,
                        beta = beta,
                        speed                  = vf,
                        coefficient_file       = c_dir + "single_heights_model_coefficients.csv",
                        min_max_file           = c_dir + "single_heights_xyz_min_max.csv")

    k_ratios = evaluate(alpha_beta_combination = alpha_beta_combination,
                        alpha = alpha,
                        beta = beta,
                        speed                  = vf,
                        coefficient_file       = c_dir + "k_ratios_model_coefficients.csv",
                        min_max_file           = c_dir + "k_ratios_xyz_min_max.csv")

    return h_single + ((number_of_layers-1) * h_single * k_ratios)


if __name__ == "__main__":

    # just using the model ---
    print(calculate_multi_layer_height(number_of_layers=1, alpha_beta_combination=3, alpha=90, beta=55, vf=300))

    # training the model yourself ----
    c, m, (x, y, z) = train("training_set/output.xlsx", "single_heights")

    # Create a grid of points for visualization
    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    zz = function((xx, yy), *c)

    # Plot the data and the fitted surface
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.plot_surface(xx, yy, zz, cmap='coolwarm')
    ax.set_title("reduced model")

    plt.show()




