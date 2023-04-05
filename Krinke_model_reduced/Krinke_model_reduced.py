import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt
import string
import random
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline


def read_training_data(filename="output.xlsx"):
    # read training data
    df = pd.read_excel("training_set/output.xlsx")

    # get a grid of the x and y values from the training data
    x    = np.linspace(1, 13, 13)
    x2   = list(zip(list(df["α [°]"]), list(df["β [°]"])))
    x2   = [x for i, x in enumerate(x2) if x not in x2[:i]]
    y    = list(df["vf [mm/min]"].unique())

    # loop over the meshgrid of x and y values and find the corresponding z values
    z_ratio, z_singleheight, z_doubleheight = [], [], []
    for [alpha, beta] in x2:
        for speed in y:
            selected_row = df[(df['α [°]'] == alpha) & (df['β [°]'] == beta) & (df['vf [mm/min]'] == speed)]
            z_ratio.append(list(selected_row['Verhältnis [-]'])[0])
            z_singleheight.append(list(selected_row['höhe einf. [mm]'])[0])
            z_doubleheight.append(list(selected_row['höhe dopp. [mm]'])[0])

    Z_exp_ratios = np.array(z_ratio).reshape(13, 8).T
    Z_exp_singleheights = np.array(z_singleheight).reshape(13, 8).T
    Z_exp_doubleheights = np.array(z_doubleheight).reshape(13, 8).T

    Z = Z_exp_singleheights

    return x, y, Z


def train(omit_training_data_indices=[]):

    x, y, Z = read_training_data()
    X, Y = np.meshgrid(x, y)

    x_flat = X.flatten()
    y_flat = Y.flatten()
    z_flat = Z.flatten()

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
    # print the results using a loop
    # print("Fit parameters:")
    # for i, name in enumerate(
    #         ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10']):
    #     print(f"{name} = {xpopt[i]:.4f} +/- {xperr[i]:.4f}")
    # # calculate the average standard error
    # avg_perr = np.mean(xperr)
    # print(f"\nAverage standard error: {avg_perr:.4f}")

    dfc = pd.DataFrame(xpopt, columns=["popt"], index=[char for char in string.ascii_lowercase][0:10])
    dfc.to_csv("model_coefficients.csv")

    dfm = pd.DataFrame([[x_min, x_max], [y_min, y_max], [z_min, z_max]], columns=["min", "max"], index=["x", "y", "z"])
    dfm.to_csv("xyz_min_max.csv")

    return xpopt, dfm, (x, y, z)


def function(X, a, b, c, d, e, f, g, h, i, j):
    x, y = X
    return a + b * x + c * y + d * x ** 2 + e * x * y + f * y ** 2 + g * x ** 3 + h * x ** 2 * y + i * x * y ** 2 + j * y ** 3


def normalize_data(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)


def denormalize_data(norm_data, min_val, max_val):
    return norm_data * (max_val - min_val) + min_val


def evaluate(alpha_beta_combination, speed):
    # read the trained model values
    coefficients = pd.read_csv("model_coefficients.csv").iloc[:, 1]
    x_min, y_min, z_min = pd.read_csv("xyz_min_max.csv").iloc[:, 1]
    x_max, y_max, z_max = pd.read_csv("xyz_min_max.csv").iloc[:, 2]

    # normalize x,y input data
    x = normalize_data(alpha_beta_combination, x_min, x_max)
    y = normalize_data(speed, y_min, y_max)

    # evaluate model output
    output = function((x, y), *coefficients)

    # denormalize model output z data
    output_norm = denormalize_data(output, z_min, z_max)

    return output_norm


def evaluate_model_performance():
    x, y, Z = read_training_data()

    z_singleheight = []
    for alpha_beta_combination in x:
        for speed in y:
            z_singleheight.append(evaluate(alpha_beta_combination, speed))

    Z_new_model = np.array(z_singleheight).reshape(13, 8).T
    # print(pd.DataFrame(Z_new_model))

    Z_difference_models_new_to_experiment = np.absolute(Z - Z_new_model)
    # print(pd.DataFrame(Z_difference_models_new_to_experiment))
    avg_z_dev, max_z_dev = np.average(Z_difference_models_new_to_experiment), np.max(Z_difference_models_new_to_experiment)
    print("(average:", avg_z_dev, ", maximum:", max_z_dev, ")")

    return avg_z_dev, max_z_dev


if __name__ == "__main__":

    # test how many points you can remove from the training set before losing performance
    n_samples = 250
    mark = [0.05, 0.1, 0.15, 0.25]
    samples = []
    max_points_removed = 95
    best_indices = [[[], 1] for _ in range(max_points_removed)]
    a = 1
    jrange = np.array(range(0, max_points_removed))[0::2]

    import os
    if not os.path.isfile(str(n_samples)+"_averaged.npy"):

        for i in range(0, n_samples):
            samples.append([])
            print(i)

            for j in jrange:
                indices_to_remove = random.sample(range(104), j)
                train(indices_to_remove)
                a, m = evaluate_model_performance()
                samples[-1].append([a, m])

                if a <= best_indices[j][1]:
                    best_indices[j] = [indices_to_remove, a]

                best_indices[j][1] = a

        samples = np.array(samples)

        print(best_indices[-1])
        print(len(best_indices[-1]))

        # across all samples, find the minimum performance loss
        performances_optimal = np.amin(samples, axis=0)
        np.save(str(n_samples) + "_optimal.npy", performances_optimal)

        # across all samples, calculate the average performance loss
        performances_sum = np.sum(samples, axis=0)
        performances_averaged = np.divide(performances_sum, n_samples)
        np.save(str(n_samples)+"_averaged.npy", performances_averaged)

        # across all samples, save the removed points that performed the best
        #np.save(str(n_samples) + "_best_indices.npy", best_indices)

    performances_averaged = np.load(str(n_samples)+"_averaged.npy")
    performances_optimal = np.load(str(n_samples)+"_optimal.npy")
    #best_indices = np.load(str(n_samples) + "_best_indices.npy")

    # cs0 = CubicSpline(jrange, performances_averaged[:, 0])
    # cs1 = CubicSpline(jrange, performances_averaged[:, 1])
    # print("minimal avg. perrformance loss:", performances_averaged[0, 0], "^= 100%")
    # print("minimal max. perrformance loss:", performances_averaged[0, 1], "^= 100%")

    # for marker in mark:
    #     y0, y1 = (1+marker)*performances_averaged[0, 0], (1+marker)*performances_averaged[0, 1]
    #     print(str(marker * 100) + "%", "reduced avg. performance ^=", y0)
    #     print(str(marker * 100) + "%", "reduced max. performance ^=", y1)
    #     #plt.axvline(x=np.interp(y0, performances_averaged[:, 0], jrange))
    #     x = np.interp(y1, performances_averaged[:, 1], jrange)
    #     plt.axvline(x=x, ymin=np.interp(x, jrange, performances_averaged[:, 1]))

    plt.xlabel("Anzahl zufällig gelöschter Messungen [x/104]")
    plt.ylabel("Durchschnittliche Abweichung (Modell zu Messung) [mm]")
    plt.title("Modellperformance bei weniger Messungen (" + str(n_samples) + " samples)")
    plt.plot(jrange, performances_averaged[:, 0], label="new model avg. deviation")
    plt.plot(jrange, performances_averaged[:, 1], label="new model max. deviation")
    plt.plot(jrange, performances_optimal[:, 0], label="new model best configuration for avg. deviation")
    plt.plot(jrange, performances_optimal[:, 1], label="new model best configuration for max. deviation")
    plt.scatter(0, 0.044, label="krinke model avg. deviation")
    plt.scatter(0, 0.170, label="krinke model max. deviation")
    plt.yticks(np.arange(0, 0.8, 0.04))
    plt.legend()
    plt.grid()
    # #plt.show()

    # # test which points are most important for model performance
    # residual_points = 19
    # samples_per_point = 100
    # Z_model_performance_avg = []
    # Z_model_performance_max = []
    # for point_index in range(104):
    #     print("--- Keeping point nr.", point_index, " and ", residual_points, "other random points ---")
    #     result = [0, 0]
    #
    #     for index, sample in enumerate(range(samples_per_point)):
    #
    #         chosen_residuals = random.sample(range(104), residual_points)
    #         while point_index in chosen_residuals:
    #             chosen_residuals = random.sample(range(104), residual_points)
    #
    #         chosen_points  = chosen_residuals + [point_index]
    #         points_to_omit = [point for point in range(104) if point not in chosen_points]
    #
    #         train(points_to_omit)
    #
    #         result = np.add(result, evaluate_model_performance())
    #         print(result/(index+1))
    #
    #     Z_model_performance_avg.append(result[0]/samples_per_point)
    #     Z_model_performance_max.append(result[1]/samples_per_point)
    #
    # Z_model_performance_avg = np.array(Z_model_performance_avg).reshape(13, 8).T
    # Z_model_performance_max = np.array(Z_model_performance_max).reshape(13, 8).T
    #
    # # Create a grid of points for visualization
    # xx, yy = np.meshgrid(np.linspace(0, 1, 13), np.linspace(0, 1, 8))
    # zz = Z_model_performance_avg
    #
    # # Plot the data and the fitted surface
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_surface(xx, yy, zz, cmap='coolwarm')

    # cmap = plt.get_cmap('Spectral')
    # c = ax.contourf(xx, yy, Z_model_performance_avg, levels=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], cmap=cmap,
    #                 corner_mask=False)

    # plt.show()

    c, m, (x, y, z) = train(best_indices[-21][0])

    # Create a grid of points for visualization
    xx, yy = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
    zz = function((xx, yy), *c)

    # Plot the data and the fitted surface
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.plot_surface(xx, yy, zz, cmap='coolwarm')
    ax.set_title("Model performance on "+ str(21+9)+ " points")

    plt.show()




