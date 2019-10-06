"""
Tensorflow 2.0 deep neural network model to predict whether more than 2.5 goals will be scored in a given football
(soccer) match.

For the purpose of this model, only the English Premier and Championship League data will be used.
"""

# Imports:
import datetime
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from data_pipeline import arrays_to_dataset, csv_to_dataframe, features_to_densefeatures
from gpu_limiter import gpu_limiter
from sklearn.model_selection import train_test_split

pd.options.display.max_rows = 500  # Set up display options for Pandas Dataframes.

if tf.test.is_gpu_available():  # Limit GPU memory growth to prevent crashing if available.
    gpu_limiter()

# Import & clean data:
df = csv_to_dataframe("./csv_files")
df['Date'] = pd.to_datetime(df['Date'])
df = df[['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].dropna().set_index('Date').sort_index()
df['Over_2.5'] = (df['FTHG'] + df['FTAG'] > 2.5).astype(int)
df = df.drop(['FTHG', 'FTAG'], axis=1)

target_column = 'Over_2.5'

# Split the data into training, validation & testing sets:;
train, test = train_test_split(df, test_size=0.2, random_state=11)
train, val = train_test_split(train, test_size=0.2, random_state=11)

# Check whether or not data is balanced:
print(train['Over_2.5'].value_counts())

# Convert features to a Tensor of DenseFeatures:
train_features = features_to_densefeatures(dataframe=train, target_column=target_column)
val_features = features_to_densefeatures(dataframe=val, target_column=target_column)
test_features = features_to_densefeatures(dataframe=test, target_column=target_column)

# Convert DenseFeature Tensors to TensorFlow Datasets:
train_ds = arrays_to_dataset(features=train_features, target=train[target_column].values, shuffle=True, batch_size=256)
val_ds = arrays_to_dataset(features=val_features, target=val[target_column].values, shuffle=False, batch_size=256)
test_ds = arrays_to_dataset(features=test_features, target=test[target_column].values, shuffle=False, batch_size=256)

# Function to build tf.keras functional API model:
def model_construct(input_shape: tuple):
    """
    Function to build a tf.keras functional API model.
    :param input_shape: Tuple of input shape. Note: shape must explicitly state that batch size is unknown
    i.e. (32,)
    :return: tf.keras model.
    """
    input_layer = tf.keras.Input(shape=input_shape, name='feature_layer')

    model = tf.keras.layers.Dense(units=8, activation='relu', name='dense_layer_0')(input_layer)
    model = tf.keras.layers.BatchNormalization(name='BN0')(model)

    model = tf.keras.layers.Dense(units=8, activation='relu', name='dense_layer_1')(model)
    model = tf.keras.layers.Dropout(rate=0.4)(model)
    model = tf.keras.layers.BatchNormalization(name='BN1')(model)

    output_layer = tf.keras.layers.Dense(units=2, activation='softmax', name='output_layer')(model)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='Model')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    return model


# Create a TensorBoard log file (time appended) directory for every run of the model:
directory = ".\\logs\\" + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.mkdir(directory)

# Create a TensorBoard callback to log a record of model performance for every 1 epoch:
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=directory,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Run "tensorboard --logdir .\logs" in anaconda prompt to review & compare logged results.
# Note: Make sure that the correct environment is activated before running.

# Create a ModelCheckpoint callback to save the best model after each epoch:
modelcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=(".\\checkpoints\\" + str('Epoch_{epoch}.h5')),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch'
)

# Build the model:
model = model_construct(input_shape=(train_features.shape[1],))
print(model.summary())

# Fit the model to the training data, using the validation data to reduce the likelihood of overfitting:
epochs = 20

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[tensorboard_callback, modelcheckpoint_callback]
)

# Save model:
model.save(".\\saved_models\\" + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + ".h5")

# Load saved model:
# loaded_model = tf.keras.models.load_model(".\\saved_models\\x.h5")  # Replace x with model to be loaded.

# Evaluate model against test data:
model_evaluation = model.evaluate(test_ds)

# Form predictions using the model:
division = str(input('Division? ') + '\n')
hometeam = str(input('Home Team? ') + '\n')
awayteam = str(input('Away Team? ') + '\n')

pred_df = pd.DataFrame({'Div': [division], 'HomeTeam':[hometeam], 'AwayTeam': [awayteam]})
pred_features = features_to_densefeatures(dataframe=pred_df, target_column=None)
prediction = model.predict(pred_features)
print('Over 2.5 Goals' if np.argmax(prediction) == 1 else 'Under 2.5 Goals')
