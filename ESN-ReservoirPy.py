import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from reservoirpy.nodes import Reservoir, Ridge, FORCE, ESN
from reservoirpy.observables import rmse, rsquare
from reservoirpy.datasets import mackey_glass

### ReservoirPy

# units - pocet neuronov v rezervoari
# lr - leaking rate - abs. hodn. najvacsieho vlastneho cisla
# sr - spectral radius -
# rc_connectivity = 1 - sparsity, default 0.1

# novy dataset
X = mackey_glass(n_timesteps=2000)

# nakreslime input
plt.figure(figsize=(10, 3))
plt.plot(X)
plt.show()

# treba definovat rezervoar a vyslednu vrstvu
reservoir = Reservoir(units=100, lr=0.3, sr=1.25)
readout = Ridge(output_dim=1, ridge=1e-5)

# vytvorime ESN
esn = reservoir >> readout

# natrenujeme model
esn.fit(X[:500], X[1:501], warmup=100)

# urobime predikcie
predictions = esn.run(X[501:-1])
#print("predictions:", predictions)

# vyhodnotime vykon modelu
print("RMSE:", rmse(X[502:], predictions), "R^2 score:", rsquare(X[502:], predictions))

