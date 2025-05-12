

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pymdp import utils
    return mo, np, plt, sns, utils


@app.cell
def _(mo):
    mo.md("## Utility Functions")
    return


@app.cell
def _(np, plt, sns):
    def plot_likelihood(matrix, xlabels = list(range(9)), ylabels = list(range(9)), title_str = "Likelihood distribution (A)"):
        """
        Plots a 2-D likelihood matrix as a heatmap
        """

        if not np.isclose(matrix.sum(axis=0), 1.0).all():
          raise ValueError("Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")
    
        fig = plt.figure(figsize = (6,6))
        ax = sns.heatmap(matrix, xticklabels = xlabels, yticklabels = ylabels, cmap = 'gray', cbar = False, vmin = 0.0, vmax = 1.0)
        plt.title(title_str)
        plt.show()

    def plot_grid(grid_locations, num_x = 3, num_y = 3 ):
        """
        Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate 
        labeled with its linear index (its `state id`)
        """

        grid_heatmap = np.zeros((num_x, num_y))
        for linear_idx, location in enumerate(grid_locations):
          y, x = location
          grid_heatmap[y, x] = linear_idx
        sns.set(font_scale=1.5)
        sns.heatmap(grid_heatmap, annot=True, cbar = False, fmt='.0f', cmap='crest')

    def plot_point_on_grid(state_vector, grid_locations):
        """
        Plots the current location of the agent on the grid world
        """
        state_index = np.where(state_vector)[0][0]
        y, x = grid_locations[state_index]
        grid_heatmap = np.zeros((3,3))
        grid_heatmap[y,x] = 1.0
        sns.heatmap(grid_heatmap, cbar = False, fmt='.0f')

    def plot_beliefs(belief_dist, title_str=""):
        """
        Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
        """

        if not np.isclose(belief_dist.sum(), 1.0):
          raise ValueError("Distribution not normalized! Please normalize")

        plt.grid(zorder=0)
        plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
        plt.xticks(range(belief_dist.shape[0]))
        plt.title(title_str)
        plt.show()
    return (plot_beliefs,)


@app.cell
def _(mo):
    mo.md("## The Basics: Categorical Distributions")
    return


@app.cell
def _(np, utils):
    my_categorical = np.random.rand(3)
    my_categorical = utils.norm_dist(my_categorical)

    print(my_categorical.reshape(-1, 1))
    print(f"Integral of distribution: {round(my_categorical.sum(), 2)}")
    return (my_categorical,)


@app.cell
def _(my_categorical, utils):
    sampled_outcome = utils.sample(my_categorical)
    print(f"Sampled outcome: {sampled_outcome}")
    return


@app.cell
def _(my_categorical, plot_beliefs):
    plot_beliefs(my_categorical, title_str = "A random (unconditional) Categorical distribution")
    return


@app.cell
def _(mo):
    mo.md("### Conditional categorical distributions")
    return


@app.cell
def _(np, utils):
    # initialize it with random numbers
    p_x_given_y = np.random.rand(3, 4)
    # Normalize it
    p_x_given_y = utils.norm_dist(p_x_given_y)
    print(p_x_given_y.round(3))
    return (p_x_given_y,)


@app.cell
def _(p_x_given_y):
    print(p_x_given_y[:, 0].reshape(-1, 1))
    print(f"Integral of P(X|Y=0) = {p_x_given_y[:, 0].sum()}")
    return


if __name__ == "__main__":
    app.run()
