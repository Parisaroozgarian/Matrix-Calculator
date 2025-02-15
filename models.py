from datetime import datetime
from app import db

class MatrixCalculation(db.Model):
    __tablename__ = 'matrix_calculation'

    id = db.Column(db.Integer, primary_key=True)
    matrix1 = db.Column(db.JSON, nullable=False)  # Store matrix as JSON
    matrix2 = db.Column(db.JSON, nullable=True)   # Optional for single matrix operations
    scalar = db.Column(db.Float, nullable=True)   # For scalar operations
    operation = db.Column(db.String(50), nullable=False)
    result = db.Column(db.JSON, nullable=False)   # Store result as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<MatrixCalculation {self.operation} at {self.created_at}>'