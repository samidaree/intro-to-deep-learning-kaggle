from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from IPython.display import display
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


""" model = keras.Sequential([
    layer.Dense(units= 10, input_shape = [2], activation = 'relu'),
    layer.Dense(units = 3, activation = "relu"), 
    layer.Dense(units= 1) 
])

model.compile(
    optimizer = 'adam',
    loss = 'mae'
) """

red_wine = pd.read_csv('./input/red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Scale to [0, 1]
max_ = df_train.max(axis=0)
min = df_train.min(axis=0)
df_train = (df_train - min) / (max_ - min)
df_valid = (df_valid - min) / (max_ - min)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']


red_wine_model = keras.Sequential([
    layers.Dense(units = 512, input_shape = [11], activation = 'relu'),
    layers.Dense(units = 512, activation = 'relu'), 
    layers.Dense(units= 512, activation = 'relu'),
    layers.Dense(units= 1)
])

red_wine_model.compile (
    optimizer = 'adam',
    loss = 'mae'
)

history = red_wine_model.fit(X_train, y_train, validation_data = (X_valid,y_valid), batch_size = 256, epochs = 10)

# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();

## Exercise 

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

fuel = pd.read_csv('./input/fuel.csv')

X = fuel.copy()
# Remove target
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)


X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mae',
)

# Feed the optimizer 128 rows of the training data at a time
model.fit(X,y,validation_data=(X,y), batch_size=128, epochs=50)
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot();