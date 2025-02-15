# Matrix Calculator

A web-based matrix calculator built with Flask and Bootstrap that performs various matrix operations. This application allows users to perform complex matrix calculations through an intuitive user interface.

## Features

- Basic Matrix Operations:
  - Addition
  - Subtraction
  - Element-wise multiplication
  - Matrix multiplication

- Advanced Operations:
  - Matrix transpose
  - Scalar multiplication
  - Determinant calculation
  - Matrix inverse
  - Eigenvalues
  - Eigenvectors

- User Interface:
  - Responsive Bootstrap design
  - Interactive form inputs
  - Clear error messages
  - Result display with print option
  - Built-in documentation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/matrix-calculator.git
cd matrix-calculator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python MyList1.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter your matrices in the correct format:
- Use nested lists with square brackets: `[[1, 2], [3, 4]]`
- Separate elements with commas
- Ensure all rows have the same number of columns

## Project Structure

```
matrix-calculator/
├── MyList1.py           # Main Flask application and matrix operations
├── requirements.txt     # Python dependencies
├── static/
│   └── style.css       # CSS styles (unused - using Bootstrap)
└── templates/
    └── index.html      # Main HTML template
```

## Dependencies

- Flask
- NumPy
- Bootstrap 5.3.2
- Bootstrap Icons

## Technical Details

### Matrix Class Implementation

The `MyList` class provides matrix operations with the following features:

- Proper dimension checking for all operations
- Comprehensive error handling
- Support for floating-point numbers
- NumPy integration for advanced calculations

### Error Handling

The calculator includes robust error handling for:
- Invalid matrix formats
- Dimension mismatches
- Singular matrices
- Non-numeric inputs
- Missing required inputs

## Browser Compatibility

Tested and working on:
- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask for the web framework
- Bootstrap for the UI components
- NumPy for matrix operations
- The open-source community for inspiration and resources
