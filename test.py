"""
Test PyTensor Knowledge Graph Embedding Models
Converted from Jupyter Notebook
"""

import numpy as np
import sys
sys.path.append("..")
from efe.models_pytensor import DistMult_Model, Complex_Model


def main():
    # ---------------------------------------------------------------------
    # Create Small Sample Data
    # ---------------------------------------------------------------------
    train_triples = np.array([
        [0, 0, 1],  # Entity 0 --relation 0--> Entity 1
        [1, 0, 2],  # Entity 1 --relation 0--> Entity 2
        [2, 1, 0],  # Entity 2 --relation 1--> Entity 0
        [0, 1, 2],  # Entity 0 --relation 1--> Entity 2
    ], dtype=np.int64)

    test_triples = np.array([
        [0, 0, 1],
        [1, 1, 2],
    ], dtype=np.int64)

    print(f"Train triples: {len(train_triples)}")
    print(f"Test  triples: {len(test_triples)}")

    # ---------------------------------------------------------------------
    # Test DistMult Model
    # ---------------------------------------------------------------------
    model = DistMult_Model()

    model.n = 3  # entities
    model.m = 2  # relations
    model.l = 3  # objects
    model.k = 5  # embedding size

    model.allocate_params()
    model.define_loss()
    model.compile_prediction()

    print(f"\nModel: {model.name}")
    print(f"Number of parameters: {model.nb_params}")

    predictions = model.predict(test_triples)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    # ---------------------------------------------------------------------
    # Test Complex Model
    # ---------------------------------------------------------------------
    complex_model = Complex_Model()
    complex_model.n = 3
    complex_model.m = 2
    complex_model.l = 3
    complex_model.k = 5

    complex_model.allocate_params()
    complex_model.define_loss()
    complex_model.compile_prediction()

    print(f"\nModel: {complex_model.name}")
    print(f"Number of parameters: {complex_model.nb_params}")

    complex_predictions = complex_model.predict(test_triples)
    print(f"Complex predictions shape: {complex_predictions.shape}")
    print(f"Complex predictions: {complex_predictions}")

    # ---------------------------------------------------------------------
    # Verify Embedding Shapes
    # ---------------------------------------------------------------------
    print("\nDistMult embeddings:")
    print(f"  Entity embeddings (e): {model.e.get_value().shape}")
    print(f"  Relation embeddings (r): {model.r.get_value().shape}")

    print("\nComplex embeddings:")
    print(f"  Entity real (e1): {complex_model.e1.get_value().shape}")
    print(f"  Entity imag (e2): {complex_model.e2.get_value().shape}")
    print(f"  Relation real (r1): {complex_model.r1.get_value().shape}")
    print(f"  Relation imag (r2): {complex_model.r2.get_value().shape}")


if __name__ == "__main__":
    main()
