# tests/test_model.py

import pytest
from src.train import train_model
from src.predict import predict
import numpy as np

def test_train_model():
    model = train_model()
    assert model is not None

def test_predict():
    model = train_model()
    example_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = predict(model, example_input)
    assert prediction is not None
