# src/predict.py

import pickle
import numpy as np


def predict(model, input_data):
    # Assuming input_data is a numpy array
    return model.predict(input_data)


if __name__ == "__main__":
    # Load the model
    with open('models/iris_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Example input data
    example_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    # Make prediction
    prediction = predict(model, example_input)
    print(f"Prediction: {prediction}")