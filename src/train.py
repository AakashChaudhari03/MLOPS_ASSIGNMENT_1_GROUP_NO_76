# src/train.py

import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Save the model
    with open('models/iris_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    return model


if __name__ == "__main__":
    train_model()
