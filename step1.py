import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd

def tutorial_keras_linear_unit() : 
    # Create a network with 1 linear unit
    model = keras.Sequential([
        layers.Dense(units=1, input_shape=[3])
    ])


def exercise_linear_model():
   # Set Matplotlib defaults
    plt.rc('figure', autolayout=True)
    plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)

    red_wine = pd.read_csv('input/red-wine.csv')
    print(red_wine.head())
    rows,columns = red_wine.shape
    red_wine_model = keras.Sequential([
        layers.Dense(1, input_shape=[columns-1])
    ])

    # random weights before training example
    model = keras.Sequential([
            layers.Dense(units=1, input_shape=[1]) 
    ])
    w,b = model.weights
    print("Weights\n{}\n\nBias\n{}".format(w,b))
    x = tf.linspace(-1.0, 1.0, 100)
    y = model.predict(x)

    plt.figure(dpi=100)
    plt.plot(x, y, 'k')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("Input: x")
    plt.ylabel("Target y")
    w, b = model.weights # you could also use model.get_weights() here
    plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
    plt.show()


exercise_linear_model()