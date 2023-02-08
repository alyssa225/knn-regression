import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from src import PolynomialRegression, KNearestNeighbor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def visualize_model(data, model, title):
    """
    This function produces a plot with the training and test datasets,
    as well as the predictions of the trained model. The plot is saved
    to `free_response/` as a png.

    Note: You should not need to change this function!

    Args:
        data (tuple of np.ndarray): four arrays containing, in order:
            training data, test data, training targets, test targets
        model: the model with a .predict() method
        title: the title for the figure

    Returns:
        train_mse (float): mean squared error on training data
        test_mse (float): mean squared error on test data
    """

    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    x_func = np.arange(-1.2, 1.2, 0.01).reshape(-1, 1)
    preds = model.predict(x_func)

    train_mse = mean_squared_error(
        y_train, model.predict(X_train))
    test_mse = mean_squared_error(
        y_test, model.predict(X_test))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)

    fig.suptitle(title)
    ax1.set_title(f"Train MSE: {train_mse:.2f}",
                  fontdict={"fontsize": "medium"})
    ax1.set_ylim(-20, 20)
    ax1.scatter(X_train, y_train)
    ax1.plot(x_func, preds, "orange", label="h(X)")
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.legend()

    ax2.set_title(f"Test MSE: {test_mse:.2f}",
                  fontdict={"fontsize": "medium"})
    ax2.set_ylim(-20, 20)
    ax2.scatter(X_test, y_test)
    ax2.plot(x_func, preds, "orange", label="h(X)")
    ax2.set_xlabel('X')
    ax2.legend()
    plt.savefig(f"free_response/{title}.png")

    return train_mse, test_mse


def part_a_plot(a, xaxis, train, test, title, save, axisname):
    """
    This uses matplotlib to create an example plot that you can modify
    for your answers to FRQ1 part a.
    """

    # Create a plot with four subplots
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 8), sharex=True)

    fig.suptitle(title)

    # One subplot for each amount of data
    amounts = a

    # Each plot has four points on the X-axis, either for `degree` or `k`
    x_axis = xaxis

    for idx, amount in enumerate(amounts):
        axes[idx].set_title(
            f"{amount} data points", fontdict={"fontsize": "small"}, pad=3)

        # This is some made-up data for the demo
        # You should replace this with your experimental results.
        # print('train_mse: ', train)
        # print('test_mse: ', test)
        train_mse = train[idx,:]
        test_mse = test[idx,:]
        # print('train_mse[i]: ', train[idx])
        # print('test_mse[i]: ', test[idx])

        # Plot a solid red line for train error
        axes[idx].plot(np.array(x_axis), train_mse, 'r-', label="Train")
        # Plot a dashed blue line for test error
        axes[idx].plot(np.array(x_axis), test_mse, 'b--', label="Test")

        axes[idx].set_ylabel('MSE')
        axes[idx].legend()

    axes[idx].set_xlabel(axisname)
    plt.savefig(save)


def load_frq_data(amount):
    '''
    Loads the data provided for this free-response question,
    with `amount` examples.

    Note: You should not need to change this function!

    Args:
        amount (int): the number of examples to include
        in the dataset. Should be one of 8, 16, 64, or 256

    Returns
        data (tuple of np.ndarray): four arrays containing, in order:
            training data, test data, training targets, test targets
    '''
    df = pd.read_csv(f"data/frq.{amount}.csv")
    x1 = df[["x"]].to_numpy()
    y1 = df[["y"]].to_numpy()
    
    return train_test_split(
        x1, y1, train_size=0.8, random_state=0, shuffle=False)


def polynomial_regression_experiment():
    """
    Run 16 experiments with fitting PolynomialRegression models
        of different degrees on datasets of different sizes.

    You will want to use the `load_frq_data` and `visualize_model`
        functions, and may want to add some print statements to
        collect data on the overall trends.
    """
    degrees = [1, 2, 4, 16]
    amounts = [8, 16, 64, 256]
    save = f"free_response/q1a_regression.png"
    subtitle = f"Regression Plot"
    train_mse = np.zeros((len(amounts),len(degrees)))
    test_mse = np.zeros((len(amounts),len(degrees)))
    for i,amount in enumerate(amounts):
        # print('i: ',i)
        for j,degree in enumerate(degrees):
            title = f"{degree}-degree Regression with {amount} points"
            X_train, X_test, y_train, y_test = load_frq_data(amount)
            # print('Poly X_train: ', X_train)
            # print('Poly X_test: ', X_test)
            # print('Poly y_train: ', y_train)
            # print('Poly y_test: ', y_test)
            data = (X_train, X_test, y_train, y_test)
            # print('Poly data: ', data)
            model = PolynomialRegression(degree)
            # print("visulize_model: ", visualize_model(data, model, title))
            train_mse[i,j], test_mse[i,j]=visualize_model(data, model, title)
            # print('finish Poly')
    part_a_plot(amounts, degrees, train_mse, test_mse, subtitle, save, 'degrees')   


def knn_regression_experiment():
    '''
    Run 16 experiments with fitting KNearestNeighbor models
        of different n_neighbors on datasets of different sizes.

    You will want to use the `load_frq_data` and `visualize_model`
        functions, and may want to add some print statements to
        collect data on the overall trends.

    Use Euclidean distance and Mean aggregation for all experiments.
    '''
    n_neighbors = [1, 2, 4, 8]
    amounts = [8, 16, 64, 256]
    save = f"free_response/q1a_knn.png"
    subtitle = f"K Nearest Neighbor Plot"
    train_mse = np.zeros((len(amounts),len(n_neighbors)))
    test_mse = np.zeros((len(amounts),len(n_neighbors)))
    for i,amount in enumerate(amounts):
        for j,neighbor in enumerate(n_neighbors):
            title = f"{neighbor}-NN with {amount} points"
            X_train, X_test, y_train, y_test = load_frq_data(amount)
            # print('knn X_train: ', X_train)
            # print('knn X_test: ', X_test)
            # print('knn y_train: ', y_train)
            # print('knn y_test: ', y_test)
            data = (X_train, X_test, y_train, y_test)
            # print('knn data: ', data)
            model = KNearestNeighbor(neighbor, 'euclidean', "mean")
            train_mse[i,j], test_mse[i,j] = visualize_model(data, model, title)
            # print('finish knn')
    part_a_plot(amounts, n_neighbors, train_mse, test_mse, subtitle, save, 'K')   

if __name__ == "__main__":
    polynomial_regression_experiment()
    knn_regression_experiment()
    # part_a_plot()
    plt.close('all')
