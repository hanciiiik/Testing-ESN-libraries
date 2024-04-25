import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scipy.sparse.linalg import eigsh
from easyesn import PredictionESN
from reservoirpy.datasets import mackey_glass

# data, train test split
X = mackey_glass(n_timesteps=2000)

inputData = X[:-1]
outputData = X[1:]

splitBoundary = int(0.8 * n_points)
inputTraining = inputData[:splitBoundary]
outputTraining = outputData[:splitBoundary]
inputTesting = inputData[splitBoundary:]
outputTesting = outputData[splitBoundary:]

# vytvorime ESN
esn = PredictionESN(n_input=1, n_output=1, n_reservoir=500, leakingRate=0.3, regressionParameters=[1e-2], solver="lsqr")

# natrenujeme model
esn.fit(inputTraining, outputTraining, transientTime="Auto", verbose=1)

# predikujeme
predictedOutput = esn.predict(inputTesting)

# output VS predicted output
plt.figure(figsize=(10, 4))
plt.plot(outputTesting, label="True Output")
plt.plot(predictedOutput, linestyle='--', label="ESN Prediction")
plt.legend()
plt.title("ESN Prediction vs True Output")
plt.show()

# meranie chyby
mse = np.mean((predictedOutput - outputTesting) ** 2)
print("Mean Squared Error:", mse)

