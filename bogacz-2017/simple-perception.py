import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""# Utils""")
    return


@app.function
def bayes_theorem(p_b_given_a, p_a, p_b):
    return p_b_given_a * p_a / p_b


@app.function
def integrate(func, domain, delta_x):

    integral = 0

    for x in domain:
        integral += func(x) * delta_x

    return integral


@app.cell
def _(np):
    def generate_polynomial(coefficients):

        order = len(coefficients) - 1

        if order < 0: return np.nan

        def p(x):
        
            y = 0
        
            for i in range(len(coefficients)):
                y += coefficients[i] * pow(x, len(coefficients) - i - 1)
    
            return y

        return p
    return (generate_polynomial,)


@app.cell
def _(generate_polynomial, np):
    coefficients = [ 2, 4, 0 ]
    func = generate_polynomial(coefficients)

    interval = (-10, 10)
    steps = 30
    domain = np.linspace(interval[0], interval[1], steps)

    riemann_sum = integrate(func, domain, 1 / steps)
    return domain, func, interval, riemann_sum, steps


@app.cell
def _(domain, func, interval, plt, riemann_sum, steps):
    plt.figure(figsize = (10, 4), layout = "constrained")

    plt.plot(domain, func(domain), label = r"$f(x)$", color = "black")
    plt.bar(domain, func(domain), label = r"$f(x) \cdot \Delta{x}$", color = "salmon", width = (interval[1] - interval[0]) * 1.1 / steps)

    plt.xlabel(r"x")
    plt.ylabel(r"f(x)")
    plt.title("Visualization of Riemann Sum\n" r"$\Sigma = {:.3}$".format(riemann_sum))

    plt.legend()

    plt.show()
    return


if __name__ == "__main__":
    app.run()
