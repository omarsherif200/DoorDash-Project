from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xg
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import keras
from custom_metrics import *


class regressionModels:
    def __init__(self, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.custom_metrics = customMetrics()

    def linearRegression(self, degree=1):
        polynomial_features = PolynomialFeatures(degree=degree)
        X_train_trans = polynomial_features.fit_transform(self.X_train)
        X_val_trans = polynomial_features.transform(self.X_val)
        PR = LinearRegression()
        PR.fit(X_train_trans, self.Y_train)
        Y_pred = PR.predict(X_val_trans)
        self.printResults(Y_pred)

    def randomForestRegressor(self, n_estimators=50, max_depth=5):
        regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        regr.fit(self.X_train, self.Y_train)
        Y_pred = regr.predict(self.X_val)
        self.printResults(Y_pred)

    def XGBRegressor(self, n_estimators=50, max_depth=5):
        xgb_r = xg.XGBRegressor(objective='reg:linear',
                                n_estimators=n_estimators, max_depth=max_depth, seed=123)
        xgb_r.fit(self.X_train, self.Y_train)
        Y_pred = xgb_r.predict(self.X_val)
        self.printResults(Y_pred)

    def KNNRegressor(self, n_neighbors=10):
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        neigh.fit(self.X_train, self.Y_train)
        Y_pred = neigh.predict(self.X_val)
        self.printResults(Y_pred)

    def DecisionTreeRegressor(self, max_depth=10):
        DT = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        DT.fit(self.X_train, self.Y_train)
        Y_pred = DT.predict(self.X_val)
        self.printResults(Y_pred)

    def ElasticNet(self, l1_ratio=0.5, alpha=1):
        Enet = ElasticNet(l1_ratio=l1_ratio, alpha=alpha)
        Enet.fit(self.X_train, self.Y_train)
        Y_pred = Enet.predict(self.X_val)
        self.printResults(Y_pred)

    def defineNeuralNetwork(self, input_shape=None):
        model = Sequential()
        model.add(Dense(35, input_shape=(35,), kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.8))
        model.add(Dense(32, kernel_initializer='normal', activation='relu'))
        # model.add(Dropout(0.8))
        model.add(Dense(16, kernel_initializer='normal', activation='relu'))

        # model.add(Dense(5,kernel_initializer='normal',activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model

    def NeuralNetworkRegressor(self, input_shape=None, loss='mean_squared_error', optimizer='adam',
                               file_path="toy_prediction_NN.h5"):
        model = self.defineNeuralNetwork(input_shape=input_shape)
        model.compile(loss=loss, optimizer=optimizer)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                                       mode='min', save_best_only=True)
        model.fit(self.X_train, self.Y_train, validation_data=(self.X_val, self.Y_val), epochs=50, callbacks=[model_checkpoint_callback],
                     batch_size=64)
        Y_pred = model.predict(self.X_val)
        self.printResults(Y_pred[:,0])

    def printResults(self, Y_pred):
        print("Mean Squared Error : " + str(mean_squared_error(self.Y_val, Y_pred)))
        print("Mean Absolute Error : " + str(mean_absolute_error(self.Y_val, Y_pred)))
        print("R2 Score : " + str(r2_score(self.Y_val, Y_pred)))
        self.custom_metrics.plot_accuracy_per_time(self.Y_val.to_numpy(), Y_pred)
        self.custom_metrics.plot_overestimate_underestimate_ratio(self.Y_val, Y_pred)