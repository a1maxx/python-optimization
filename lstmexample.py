import numpy as np
import matplotlib.pyplot as plt

# Generate data
n = 500
t = np.linspace(0, 20.0 * np.pi, n)
X = np.sin(t)

# Set window of past points for LSTM model
window = 10

last = int(n / 5.0)
Xtrain = X[:-last]
Xtest = X[-last - window:]

xin = []
next_X = []

for i in range(window, len(Xtrain)):
    xin.append(Xtrain[i - window:i])
    next_X.append(Xtrain[i])

# Reshape data to format for LSTM
xin, next_X = np.array(xin), np.array(next_X)
xin = xin.reshape(xin.shape[0], xin.shape[1], 1)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

m = Sequential()
m.add(LSTM(units=50, return_sequences=True, input_shape=(xin.shape[1], 1)))
m.add(Dropout(0.2))
m.add(LSTM(units=50))
m.add(Dropout(0.2))
m.add(Dense(units=1))
m.compile(optimizer='adam', loss='mean_squared_error')

history = m.fit(xin, next_X, epochs=50, batch_size=50, verbose=0)
plt.figure()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.semilogy(history.history['loss'])
plt.show()

xin = []
next_X1 = []
for i in range(window, len(Xtest)):
    xin.append(Xtest[i - window:i])
    next_X1.append(Xtest[i])

# Reshape data to format for LSTM
xin, next_X1 = np.array(xin), np.array(next_X1)
xin = xin.reshape((xin.shape[0], xin.shape[1], 1))

X_pred = m.predict(xin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred, ':', label='LSTM')
plt.plot(next_X1, '--', label='Actual')
plt.legend()
plt.show()

# Using predicted values to predict next step
X_pred = Xtest.copy()
for i in range(window, len(X_pred)):
    xin = X_pred[i - window:i].reshape((1, window, 1))
    X_pred[i] = m.predict(xin)

# Plot prediction vs actual for test data
plt.figure()
plt.plot(X_pred[window:], ':', label='LSTM')
plt.plot(next_X1, '--', label='Actual')
plt.legend()
plt.show()

# %% Digital Twin with LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

# For LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Load training data
# file = 'https://apmonitor.com/do/uploads/Main/tclab_dyn_data3.txt'
train = pd.read_csv("tclab_data3.csv")

# Scale features
s1 = MinMaxScaler(feature_range=(-1, 1))
Xs = s1.fit_transform(train[['T1', 'Q1']])

# Scale predicted value
s2 = MinMaxScaler(feature_range=(-1, 1))
Ys = s2.fit_transform(train[['T1']])

# Each time step uses last 'window' to predict the next change
window = 70
X = []
Y = []
for i in range(window, len(Xs)):
    X.append(Xs[i - window:i, :])
    Y.append(Ys[i])

# Reshape data to format accepted by LSTM
X, Y = np.array(X), np.array(Y)

# create and train LSTM model

# Initialize LSTM model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, \
               input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error', \
              metrics=['accuracy'])

# Allow for early exit
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

# Fit (and time) LSTM model
t0 = time.time()
history = model.fit(X, Y, epochs=10, batch_size=250, callbacks=[es], verbose=1)
t1 = time.time()
print('Runtime: %.2f s' % (t1 - t0))

# Plot loss
plt.figure(figsize=(8, 4))
plt.semilogy(history.history['loss'])
plt.xlabel('epoch');
plt.ylabel('loss')
plt.savefig('tclab_loss.png')
model.save('model.h5')

# Verify the fit of the model
Yp = model.predict(X)

# un-scale outputs
Yu = s2.inverse_transform(Yp)
Ym = s2.inverse_transform(Y)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(train['Time'][window:], Yu, 'r-', label='LSTM')
plt.plot(train['Time'][window:], Ym, 'k--', label='Measured')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(train['Q1'], label='heater (%)')
plt.legend()
plt.xlabel('Time (sec)');
plt.ylabel('Heater')
plt.show()
# plt.savefig('tclab_fit.png')

# Load model
v = load_model('model.h5')
# Load training data
test = pd.read_csv('tclab_data4.csv')


Xt = test[['T1', 'Q1']].values
Yt = test[['T1']].values

Xts = s1.transform(Xt)
Yts = s2.transform(Yt)

Xti = []
Yti = []
for i in range(window, len(Xts)):
    Xti.append(Xts[i - window:i, :])
    Yti.append(Yts[i])

# Reshape data to format accepted by LSTM
Xti, Yti = np.array(Xti), np.array(Yti)

# Verify the fit of the model
Ytp = model.predict(Xti)

# un-scale outputs
Ytu = s2.inverse_transform(Ytp)
Ytm = s2.inverse_transform(Yti)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(test['Time'][window:], Ytu, 'r-', label='LSTM Predicted')
plt.plot(test['Time'][window:], Ytm, 'k--', label='Measured')
plt.legend()
plt.ylabel('Temperature (°C)')
plt.subplot(2, 1, 2)
plt.plot(test['Time'], test['Q1'], 'b-', label='Heater')
plt.xlabel('Time (sec)');
plt.ylabel('Heater (%)')
plt.legend()
plt.show()

# Using predicted values to predict next step
Xtsq = Xts.copy()
for i in range(window, len(Xtsq)):
    Xin = Xtsq[i - window:i].reshape((1, window, 2))
    Xtsq[i][0] = v.predict(Xin)
    Yti[i - window] = Xtsq[i][0]

# Ytu = (Yti - s2.min_[0])/s2.scale_[0]
Ytu = s2.inverse_transform(Yti)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(test['Time'][window:], Ytu, 'r-', label='LSTM Predicted')
plt.plot(test['Time'][window:], Ytm, 'k--', label='Measured')
plt.legend()
plt.ylabel('Temperature (°C)')
plt.subplot(2, 1, 2)
plt.plot(test['Time'], test['Q1'], 'b-', label='Heater')
plt.xlabel('Time (sec)');
plt.ylabel('Heater (%)')
plt.legend()
plt.show()
