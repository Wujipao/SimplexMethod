import re
import numpy as np
from prettytable import PrettyTable
from graph import Graph

class SimplexSolver:
    def __init__(self):
        self.objective_function = None
        self.constraints = []
        self.num_variables = 0
        self.num_constraints = 0
        self.tableau = None
        self.var_names = []
        self.slack_vars = []
        self.basic_vars = []

    def get_objective_function(self): # For getting the objective function
        while True:
            try:
                self.objective_function = input("Enter the objective function (e.g., 'z = 100x + 200y'): ")
                match = re.match(r'z\s*=\s*(.+)', self.objective_function) # This ensures that the OF input starts with "z = "
                if not match:
                    raise ValueError("Invalid format. Please start with 'z =' and specify terms like '100x + 200y'.")
                # Separates coefficients, operator, and variables, and store them in a list of tuples (e.g. [('120', 'x'), ('100', 'y')]
                terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z])', match.group(1))
                self.var_names = sorted(set([term[1] for term in terms])) # Sorting variables for consistency
                self.num_variables = len(self.var_names) # Calculates the number of decision variables in OF.
                self.objective_function = [0] * self.num_variables # Initializes variables to 0
                for coef, var in terms:
                    coef = coef.replace(" ", "")  # Remove any whitespace
                    coef = float(coef) if coef not in ["", "+"] else 1.0 # Converts coefficient into float. If coefficient is empty, it will be 1.0 automatically.
                    if coef == "-": # If it is accompanied by -, assume that it is negative. Thus, -1.0.
                        coef = -1.0
                    idx = self.var_names.index(var) # Index for variables
                    self.objective_function[idx] = coef # Pass the coefficient value to OF since it was initialized as 0.
                break
            except ValueError as e:
                print(e)

    def get_constraints(self): # For getting the constraints
        while True:
            try:
                self.num_constraints = int(input("Enter the number of constraints: "))
                if self.num_constraints <= 0:
                    raise ValueError("Number of constraints must be greater than zero.") # Ensures that constraints is not less than 0.
                break
            except ValueError:
                print("Please enter a valid positive integer for the number of constraints.")

        for i in range(self.num_constraints):
            while True:
                try:
                    constraint = input(f"Enter constraint {i + 1} (e.g., '2x + 3y <= 10'): ")
                    match = re.match(r'(.+)\s*(<=|>=|=)\s*(-?\d+\.?\d*)', constraint) # Separates the left hand side and right hand side.
                    # (.+): Matches the left-hand side (e.g., 2x + 3y).
                    # \s*(<=|>=|=)\s*: Matches the operator (<=, >=, or =), allowing surrounding spaces.
                    # (-?\d+\.?\d*): Matches the right-hand side (a number, e.g., 10 or -15).
                    if not match:
                        raise ValueError("Invalid constraint format. Ensure it follows the form '2x + 3y <= 10'.")
                    self.constraints.append((match.group(1), match.group(2), float(match.group(3)))) # Extracts the separated elements
                    # match.group(1) extracts the left-hand side (e.g., 2x + 3y).
                    # match.group(2) extracts the operator (e.g., <=).
                    # match.group(3) extracts the right-hand side, converted to a float (e.g., 10.0).
                    # Then, stores them in a tuple
                    break
                except ValueError as e:
                    print(e)

    def parse_constraints(self): # For converting constraints into matrix suitable for simplex tableau
        # Some processes are the same with the objective function
        constraint_matrix = []
        rhs = []

        for i, (expr, operator, value) in enumerate(self.constraints): # Iterate through each constraint to convert into matrix form.
            # (e.g. expr = '2x + 3y', operator = '<=', value = 10.0) BTW, value is the right-hand side.
            row = [0] * self.num_variables
            terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z])', expr) # Extracting coefficients and variables
            for coef, var in terms:
                coef = coef.replace(" ", "")  # Remove any whitespace
                coef = float(coef) if coef not in ["", "+"] else 1.0
                if coef == "-":
                    coef = -1.0
                idx = self.var_names.index(var)
                row[idx] = coef

            slack_var = [0] * self.num_constraints
            if operator in ["<=", "="]:
                slack_var[i] = 1  # Add a slack variable for <= or = constraints
                self.slack_vars.append(f"S{i + 1}")
            elif operator == ">=":
                slack_var[i] = -1  # Add a surplus variable for >= constraints
                self.slack_vars.append(f"S{i + 1}")

            row.extend(slack_var)
            constraint_matrix.append(row)
            rhs.append(value)

        # Initialize basic variables with slack variables
        self.basic_vars = [f"S{i + 1}" for i in range(self.num_constraints)]

        return constraint_matrix, rhs

    def create_tableau(self, constraint_matrix, rhs):
        tableau = np.array(constraint_matrix, dtype=float)
        tableau = np.hstack((tableau, np.array(rhs).reshape(-1, 1)))

        # Add the objective function row
        obj_row = [-coef for coef in self.objective_function] + [0] * (self.num_constraints + 1)
        tableau = np.vstack((tableau, obj_row))

        return tableau

    def display_tableau(self, tableau, iteration): # For tableau printing
        table = PrettyTable()
        headers = self.var_names + self.slack_vars + ["RHS"] # Headings.
        table.field_names = ["BV"] + headers # Slack variables, decision variables, and RHS, will be added alongside "BV"

        for i, row in enumerate(tableau[:-1]):
            formatted_row = [self.format_value(val) for val in row]
            table.add_row([self.basic_vars[i]] + formatted_row) # Basic variables will be stored first before tha actual values.

        formatted_last_row = [self.format_value(val) for val in tableau[-1]]
        table.add_row(["z"] + formatted_last_row)
        print(f"\n--- Tableau at Iteration {iteration} ---")
        print(table)

    def format_value(self, value): # For conversion of 0 decimal point to integer (e.g. 1.0 to 1)
        # Format value as an integer if its decimal part is zero; otherwise, keep it as a float.
        return f"{int(value)}" if value.is_integer() else f"{value:.2f}"

    def display_lp_model(self): # For LP model printing
        print("\nLinear Programming Model")
        obj_func = f"z = {self.objective_function[0]}{self.var_names[0]}"
        for coef, var in zip(self.objective_function[1:], self.var_names[1:]):
            obj_func += f" + {coef}{var}" if coef >= 0 else f" - {abs(coef)}{var}"
        obj_func += " " + " ".join([f" + 0{sv}" for sv in self.slack_vars]) + " = 0"
        print(obj_func)
        print("Subject to constraints:")
        for i, constraint in enumerate(self.constraints):
            print(f"{constraint[0]} {constraint[1]} {constraint[2]}")
        print(f"{', '.join(self.var_names)} >= 0")

    def solve_simplex(self): # For solving the simplex method
        constraint_matrix, rhs = self.parse_constraints()
        self.tableau = self.create_tableau(constraint_matrix, rhs)

        iteration = 1
        while True:
            self.display_tableau(self.tableau, iteration)

            pivot_col = self.get_pivot_column() # Find the pivot column
            if pivot_col is None:
                print("\nOptimal solution found.")
                break

            pivot_row = self.get_pivot_row(pivot_col) # Find the pivotal row
            if pivot_row is None:
                raise ValueError("Problem is unbounded.")

            print(f"Pivotal Column: {[f'{self.tableau[i, pivot_col]:.2f}' for i in range(len(self.tableau))]}")
            print(f"Pivotal Row: {[f'{self.tableau[pivot_row, j]:.2f}' for j in range(len(self.tableau[0]))]}")
            pivot_element = self.tableau[pivot_row, pivot_col]
            print(f"Pivot: {pivot_element:.2f} or (1/{pivot_element:.2f})")
            self.perform_pivot(pivot_row, pivot_col)
            self.basic_vars[pivot_row] = self.var_names[pivot_col]  # Update basic variables
            iteration += 1

            solution = {var: 0 for var in self.var_names} # Extracting the solution
            for i, var in enumerate(self.basic_vars):
                if var in self.var_names:
                    solution[var] = self.tableau[i, -1]

        self.display_solution()
        return solution

    def get_pivot_column(self):
        last_row = self.tableau[-1, :-1] # Get the last row in the tableau
        min_value = np.min(last_row) # Find the most negative value.
        if min_value >= 0: # If there is no negative value, return none.
            return None
        return np.argmin(last_row)

    def get_pivot_row(self, pivot_col):
        column = self.tableau[:-1, pivot_col] # Extract the values from the selected pivot column, excluding the last row.
        rhs = self.tableau[:-1, -1] # Get the values in RHS column
        ratios = np.divide(rhs, column, out=np.full_like(rhs, np.inf), where=column > 0) # Get the ratios
        min_ratio = np.min(ratios) # Find the least ratio
        if min_ratio == np.inf:
            return None
        return np.argmin(ratios)

    def perform_pivot(self, pivot_row, pivot_col):
        pivot_element = self.tableau[pivot_row, pivot_col] # The intersection will be the pivot element
        self.tableau[pivot_row] /= pivot_element # Divide the values in pivot row from the pivot element to make the pivot element equal to 1.
        for i in range(len(self.tableau)):
            if i != pivot_row:
                self.tableau[i] -= self.tableau[i, pivot_col] * self.tableau[pivot_row]

    def display_solution(self):
        print("\nOptimal Solution:")
        print(f"z = {self.tableau[-1, -1]:.2f}")
        for i, var in enumerate(self.var_names + self.slack_vars):
            value = 0
            for j in range(len(self.constraints)):
                if self.tableau[j, i] == 1:
                    value = self.tableau[j, -1]
                    break
            print(f"{var} = {value:.2f}")

    def display_solution_check(self, solution): # Displays step-by-step verification of the solution against the objective function and constraints.
        print("\nSolution Check:")
        # For verifying objective function
        print("Objective Function Check:")
        z_calculated = 0
        z_expression = []
        for idx, coef in enumerate(self.objective_function):
            var_name = self.var_names[idx]
            z_calculated += coef * solution.get(var_name, 0) # Multiply the coefficient to the solution variables.
            # (e.g. coef = 120 * solution.get(var_name) = 1.5)
            z_expression.append(f"{coef}({solution.get(var_name, 0):.2f})") # Store in z_expression as string for printing.

        solution['z'] = z_calculated # Update the z value
        z_expression_str = " + ".join(z_expression)
        print(f"z = {' + '.join([f'{c}{v}' for c, v in zip(self.objective_function, self.var_names)])}")
        print(f"{solution.get('z', 0):.2f} = {z_expression_str}")
        print(f"{solution.get('z', 0):.2f} = {z_calculated:.2f}\n")

        if np.isclose(solution.get('z', 0), z_calculated):
            print("Objective function is satisfied.")
        else:
            print("Objective function is not satisfied.")

        # For verifying constraints
        print("\nConstraints Check:")

        for i, (expr, operator, value) in enumerate(self.constraints):
            terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*([a-zA-Z])', expr) # Same process in parse_constraints. Separates the coefficient and variables.
            constraint_calculated = 0
            constraint_expression = []
            for coef, var in terms:
                coef = float(coef.replace(" ", "")) if coef.strip() else 1.0
                if coef == "-":
                    coef = -1.0
                var_value = solution.get(var, 0)
                constraint_calculated += coef * var_value
                constraint_expression.append(f"{coef}({var_value:.2f})")

            constraint_expression_str = " + ".join(constraint_expression)
            print(f"Constraint {i + 1}: {expr} {operator} {value}")
            print(f"{constraint_expression_str} = {constraint_calculated:.2f}")
            print(f"{constraint_calculated:.2f} = {value}")

            # For checking if the constraints is satisfied
            if operator == "<=":
                satisfied = constraint_calculated <= value
            elif operator == ">=":
                satisfied = constraint_calculated >= value
            else:
                satisfied = np.isclose(constraint_calculated, value)

            status = "satisfied" if satisfied else "not satisfied"
            print(f"Constraint {i + 1} is {status}.\n")

    def run(self):
        self.get_objective_function()
        self.get_constraints()
        self.display_lp_model()
        try:
            solution = self.solve_simplex()
            self.display_solution_check(solution)
            graph = Graph(self.var_names, self.constraints, solution)
            graph.plot()
        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    solver = SimplexSolver()
    solver.run()