from flask import Flask, render_template, request
import numpy as np
import ast

app = Flask(__name__)

def parse_matrix(matrix_str):
    """Parse a string representation of a matrix into a MyList object."""
    try:
        matrix = ast.literal_eval(matrix_str)
        if not isinstance(matrix, (list, tuple)) or not all(isinstance(row, (list, tuple)) for row in matrix):
            raise ValueError("Matrix must be a list of lists.")
        if len(set(len(row) for row in matrix)) != 1:
            raise ValueError("All rows must have the same number of columns.")
        
        n = len(matrix)
        m = len(matrix[0])
        
        return MyList(n, m, matrix)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid matrix format: {str(e)}. Ensure you use a 2D list, e.g., [[1, 2], [3, 4]].")
    except Exception as e:
        raise ValueError(f"Error parsing matrix: {str(e)}")

class MyList:
    """A class for matrix operations with improved implementation."""
    def __init__(self, n, m, data=None):
        if data is not None:
            if not all(isinstance(x, (int, float)) for row in data for x in row):
                raise ValueError("Matrix elements must be numbers")
            self.__list = [row[:] for row in data]
            self.sizerow = len(self.__list)
            self.sizecol = len(self.__list[0]) if self.sizerow > 0 else 0
        else:
            self.__list = [[0 for _ in range(m)] for _ in range(n)]
            self.sizerow = n
            self.sizecol = m

    def set(self, i, j, x):
        if not isinstance(x, (int, float)):
            raise ValueError("Matrix elements must be numbers")
        if not (0 <= i < self.sizerow and 0 <= j < self.sizecol):
            raise IndexError("Matrix indices out of range")
        self.__list[i][j] = x

    def get(self, i, j):
        if not (0 <= i < self.sizerow and 0 <= j < self.sizecol):
            raise IndexError("Matrix indices out of range")
        return self.__list[i][j]

    def __str__(self):
        return '\n'.join(['\t'.join(map(str, row)) for row in self.__list])

    def __add__(self, m2):
        if not isinstance(m2, MyList):
            raise TypeError("Can only add two MyList matrices")
        if self.sizerow != m2.sizerow or self.sizecol != m2.sizecol:
            raise ValueError("Matrices must have the same dimensions for addition")
        result = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                result.set(i, j, self.__list[i][j] + m2.__list[i][j])
        return result

    def __sub__(self, m2):
        if not isinstance(m2, MyList):
            raise TypeError("Can only subtract two MyList matrices")
        if self.sizerow != m2.sizerow or self.sizecol != m2.sizecol:
            raise ValueError("Matrices must have the same dimensions for subtraction")
        result = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                result.set(i, j, self.__list[i][j] - m2.__list[i][j])
        return result

    def __mul__(self, m2):
        """Element-wise multiplication (Hadamard product)"""
        if not isinstance(m2, MyList):
            raise TypeError("Can only multiply two MyList matrices")
        if self.sizerow != m2.sizerow or self.sizecol != m2.sizecol:
            raise ValueError("Matrices must have the same dimensions for element-wise multiplication")
        result = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                result.set(i, j, self.__list[i][j] * m2.__list[i][j])
        return result

    def __matmul__(self, m2):
        """Matrix multiplication"""
        if not isinstance(m2, MyList):
            raise TypeError("Can only multiply two MyList matrices")
        if self.sizecol != m2.sizerow:
            raise ValueError(f"Matrix dimensions incompatible for multiplication: ({self.sizerow}x{self.sizecol}) @ ({m2.sizerow}x{m2.sizecol})")
        
        result = MyList(self.sizerow, m2.sizecol)
        for i in range(self.sizerow):
            for j in range(m2.sizecol):
                total = sum(self.__list[i][k] * m2.__list[k][j] for k in range(self.sizecol))
                result.set(i, j, total)
        return result

    def transpose(self):
        """Return the transpose of the matrix"""
        result = MyList(self.sizecol, self.sizerow)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                result.set(j, i, self.__list[i][j])
        return result

    def scalar_multiply(self, scalar):
        """Multiply the matrix by a scalar value"""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")
        result = MyList(self.sizerow, self.sizecol)
        for i in range(self.sizerow):
            for j in range(self.sizecol):
                result.set(i, j, self.__list[i][j] * scalar)
        return result

    def to_numpy(self):
        """Convert to numpy array for advanced operations"""
        return np.array(self.__list, dtype=float)

    def is_square(self):
        """Check if the matrix is square"""
        return self.sizerow == self.sizecol

    def determinant(self):
        """Calculate the determinant of the matrix"""
        if not self.is_square():
            raise ValueError("Determinant is only defined for square matrices")
        return float(np.linalg.det(self.to_numpy()))

    def inverse(self):
        """Calculate the inverse of the matrix"""
        if not self.is_square():
            raise ValueError("Matrix must be square to have an inverse")
        try:
            inv = np.linalg.inv(self.to_numpy())
            result = MyList(self.sizerow, self.sizecol)
            for i in range(self.sizerow):
                for j in range(self.sizecol):
                    result.set(i, j, float(inv[i][j]))
            return result
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular (not invertible)")

@app.route('/', methods=['GET', 'POST'])
def index():
    result_str = None
    errors = []
    
    if request.method == 'POST':
        try:
            matrix1_str = request.form.get('matrix1')
            matrix2_str = request.form.get('matrix2')
            scalar_str = request.form.get('scalar')
            operation = request.form.get('operation')

            # Initialize matrices
            matrix1 = parse_matrix(matrix1_str)
            matrix2 = parse_matrix(matrix2_str) if matrix2_str else None
            
            # Parse scalar
            scalar = float(scalar_str) if scalar_str else None

            # Perform calculation
            result = None
            if operation == "add":
                result = matrix1 + matrix2
            elif operation == "subtract":
                result = matrix1 - matrix2
            elif operation == "multiply":
                result = matrix1 * matrix2
            elif operation == "matmul":
                result = matrix1 @ matrix2
            elif operation == "transpose":
                result = matrix1.transpose()
            elif operation == "scalar_multiply":
                if scalar is None:
                    raise ValueError("Scalar value is required for scalar multiplication")
                result = matrix1.scalar_multiply(scalar)
            elif operation == "determinant":
                result = matrix1.determinant()
            elif operation == "inverse":
                result = matrix1.inverse()
            elif operation == "eigenvalues":
                if not matrix1.is_square():
                    raise ValueError("Eigenvalues can only be calculated for square matrices")
                result = np.linalg.eigvals(matrix1.to_numpy())
            elif operation == "eigenvectors":
                if not matrix1.is_square():
                    raise ValueError("Eigenvectors can only be calculated for square matrices")
                eigenvals, eigenvecs = np.linalg.eig(matrix1.to_numpy())
                # Return both eigenvalues and eigenvectors in a formatted string
                result = f"Eigenvalues:\n{eigenvals}\n\nEigenvectors:\n{eigenvecs}"

            # Format the result
            if isinstance(result, (np.ndarray, float)):
                result_str = str(result)
            elif result is not None:
                result_str = str(result)

        except Exception as e:
            errors.append(str(e))

    return render_template('index.html', 
                         result=result_str, 
                         errors=errors)

if __name__ == '__main__':
    app.run(debug=True)