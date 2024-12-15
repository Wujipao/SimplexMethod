import re
import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, var_names, constraints, solution):
        self.var_names = var_names
        self.constraints = constraints
        self.solution = solution

    def plot(self):
        x = np.linspace(0, 10, 400) # Creates 400 equally spaced points for the x variable between 0 and 10.
        for expr, operator, value in self.constraints: # Iterates the constraints
            terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z])', expr) # Same process; separation of elements.
            if len(terms) == 2:
                coef1, var1 = terms[0]
                coef2, var2 = terms[1]
                coef1 = float(coef1.replace(" ", "")) if coef1.strip() else 1.0 # Converts the coefficient into numeric values
                coef2 = float(coef2.replace(" ", "")) if coef2.strip() else 1.0
                if coef1 == "-": coef1 = -1.0
                if coef2 == "-": coef2 = -1.0
                if var1 == self.var_names[0]: # Rearrange the constraint into slope-intercept form
                    plt.plot(x, (value - coef1 * x) / coef2, label=f'{expr} {operator} {value}')
                else:
                    plt.plot((value - coef2 * x) / coef1, x, label=f'{expr} {operator} {value}')

        plt.xlim((0, 10))
        plt.ylim((0, 10))
        plt.xlabel(self.var_names[0]) # Labels the axes with variable names (x and y).
        plt.ylabel(self.var_names[1])
        plt.legend()
        plt.grid(True)

        # Plot the solution point
        plt.scatter(self.solution.get(self.var_names[0], 0), self.solution.get(self.var_names[1], 0), color='red', zorder=5)
        plt.title('Linear Programming Solution Graph')
        plt.show()
