from logisticRegression import build_model
from cnn import build_cnn
from keras.datasets import mnist
import pytest
from keras.utils import to_categorical

def test_logisticRegression():
    # Test that the model makes correct predictions
    X_test = [[5.0, 3.0, 1.5, 0.2], [6.0, 3.0, 4.5, 1.5], [7.0, 3.2, 4.7, 1.4], [6.7, 3.0, 5.2, 2.3]]
    y_pred = build_model()[0].predict(X_test)
    
    assert list(y_pred) == [0, 1, 1, 2]

def test_cnn():
   cnn_model = build_cnn()
   _, (X_test, y_test) = mnist.load_data()
   y_test = to_categorical(y_test)

    # evaluate the model
   _, test_acc = cnn_model.evaluate(X_test, y_test)
    
    # check if accuracy is greater than 0.9
   assert test_acc > 0.9