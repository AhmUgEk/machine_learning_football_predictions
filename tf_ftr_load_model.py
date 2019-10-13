"""
Example of how to load a saved model and form predictions without having to retrain.
"""

# Imports:
import numpy as np

import pandas as pd
import tensorflow as tf
from data_pipeline import features_to_densefeatures

# Load model:
model = tf.keras.models.load_model(".\\saved_models\\20191013-170945.h5")

# Form predictions using the model:
division = str(input('Division? ') + '\n')
hometeam = str(input('Home Team? ') + '\n')
awayteam = str(input('Away Team? ') + '\n')
b365h = float(input('Bet 365 Home Win Odds? ') + '\n')
b365d = float(input('Bet 365 Draw Odds? ') + '\n')
b365a = float(input('Bet 365 Away Win Odds? ') + '\n')
whh = float(input('William Hill Home Win Odds? ') + '\n')
whd = float(input('William Hill Draw Odds? ') + '\n')
wha = float(input('William Hill Away Win Odds? ') + '\n')

# Form predictions using trained model:
pred_df = pd.DataFrame({
    'Div': [division],
    'HomeTeam': [hometeam],
    'AwayTeam': [awayteam],
    'B365H': [b365h],
    'B365D': [b365d],
    'B365A': [b365a],
    'WHH': [whh],
    'WHD': [whd],
    'WHA': [wha]
})

pred_features = features_to_densefeatures(dataframe=pred_df, target_column=None)
pred_ds = tf.data.Dataset.from_tensor_slices(np.array(pred_features))
prediction = model.predict(pred_features)

print(f"\nLikelihood of Away Team Win = {prediction[0][0]*100:.2f}%\n")
print(f"Likelihood of Draw = {prediction[0][1]*100:.2f}%\n")
print(f"Likelihood of Home Team Win = {prediction[0][2]*100:.2f}%")
