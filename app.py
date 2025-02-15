from flask import Flask, render_template, request
import numpy as np
import ast
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "dev"  # For development - in production use environment variable

def parse_matrix(matrix_str):
    """Parse a string representation of a matrix into a numpy array."""
    try:
        if not matrix_str or matrix_str.isspace():
            return None
            
        # Parse string to Python list
        matrix = ast.literal_eval(matrix_str)
        
        # Validate structure
        if not isinstance(matrix, (list, tuple)) or not all(
                isinstance(row, (list, tuple)) for row in matrix):
            raise ValueError("Matrix must be a list of lists")
            
        # Convert to numpy array
        return np.array(matrix, dtype=float)
        
    except (ValueError, SyntaxError) as e:
        raise ValueError(
            f"Invalid matrix format: {str(e)}. Use format like [[1, 2], [3, 4]]")
    except Exception as e:
        raise ValueError(f"Error parsing matrix: {str(e)}")

def validate_matrices(matrix1, matrix2=None, operation=None):
    """Validate matrices based on the operation."""
    if operation in ["add", "subtract", "multiply"]:
        if matrix2 is None:
            raise ValueError("Second matrix is required for this operation")
        if matrix1.shape != matrix2.shape:
            raise ValueError(f"Matrices must have same shape for {operation}")
            
    elif operation == "matmul":
        if matrix2 is None:
            raise ValueError("Second matrix is required for matrix multiplication")
        if matrix1.shape[1] != matrix2.shape[0]:
            raise ValueError(
                f"Matrix shapes incompatible for multiplication: {matrix1.shape} and {matrix2.shape}")
                
    elif operation in ["determinant", "inverse", "eigenvalues", "eigenvectors"]:
        if matrix1.shape[0] != matrix1.shape[1]:
            raise ValueError(f"Square matrix required for {operation}")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    errors = []
    
    if request.method == 'POST':
        try:
            # Get form inputs
            matrix1_str = request.form.get('matrix1')
            matrix2_str = request.form.get('matrix2')
            scalar_str = request.form.get('scalar')
            operation = request.form.get('operation')
            
            # Parse first matrix (required for all operations)
            matrix1 = parse_matrix(matrix1_str)
            if matrix1 is None:
                raise ValueError("First matrix is required")
                
            # Parse second matrix (if provided)
            matrix2 = parse_matrix(matrix2_str) if matrix2_str else None
            
            # Parse scalar (if provided)
            scalar = float(scalar_str) if scalar_str else None
            
            # Validate matrices based on operation
            validate_matrices(matrix1, matrix2, operation)
            
            # Perform calculations
            if operation == "add":
                result = np.add(matrix1, matrix2)
            elif operation == "subtract":
                result = np.subtract(matrix1, matrix2)
            elif operation == "multiply":
                result = np.multiply(matrix1, matrix2)
            elif operation == "matmul":
                result = np.matmul(matrix1, matrix2)
            elif operation == "transpose":
                result = np.transpose(matrix1)
            elif operation == "scalar_multiply":
                if scalar is None:
                    raise ValueError("Scalar value is required for scalar multiplication")
                result = np.multiply(matrix1, scalar)
            elif operation == "determinant":
                result = np.linalg.det(matrix1)
            elif operation == "inverse":
                result = np.linalg.inv(matrix1)
            elif operation == "eigenvalues":
                result = np.linalg.eigvals(matrix1)
            elif operation == "eigenvectors":
                eigenvals, eigenvecs = np.linalg.eig(matrix1)
                result = f"Eigenvalues:\n{eigenvals}\n\nEigenvectors:\n{eigenvecs}"
                
            # Format result for display
            if isinstance(result, (np.ndarray, float, complex)):
                result = str(result)
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Error in calculation: {str(e)}")
            
    return render_template('index.html', result=result, errors=errors)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
