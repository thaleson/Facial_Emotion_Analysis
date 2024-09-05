# Script to prepare datasets, load data, and generate scaled images
import pandas as pd
import numpy as np

# Load the dataset from the CSV file
# The original dataset can be found at: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
data = pd.read_csv('data/fer2013.csv')
data = data['pixels']

# Split the pixel values and convert them to a numpy array
data = [dat.split() for dat in data]
data = np.array(data)
data = data.astype('float64')

# Normalize the pixel values to the range [0, 1]
data = [[np.divide(d, 255.0) for d in dat] for dat in data]

# Save the processed data as a numpy binary file
np.save('data/Scaled.bin.npy', data)
