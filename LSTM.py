import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
from keras.layers import Dense, LSTM, Dropout, CuDNNLSTM, TimeDistributed, Flatten, AveragePooling1D, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
import tensorflow
from keras.callbacks import CSVLogger
import seaborn as sns
from keras import regularizers
from keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import Callback, EarlyStopping


class Classifier:
    def __init__(self, path, seed=0, timesteps=128):
        self.data_dim = 0
        self.df = pd.read_csv(path, index_col=0)
        self.df_best = pd.DataFrame(columns=['seed', 'Timestep', 'Acc', 'Val_acc', 'Loss', 'Val_loss'])
        self.timesteps = timesteps
        self.seed = seed
        self.shuffle_rows()
        self.shape_data()
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.split_()
        self.model_()
        self.return_results()

    def shuffle_rows(self):
        self.df = self.df.sample(frac=1)

    def shape_data(self):
        '''shape the data to fit the LSTM model'''
        data_length = 7680 #length of each EEG data, equivalent to 12 seconds of data
        timesteps = self.timesteps
        self.data_dim = data_length // timesteps
        print('data dimension: ', self.data_dim)
        print('timesteps: ', self.timesteps)

    def split_(self):
        X = self.df.drop(['y'], axis=1)
        y = self.df['y']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, random_state=self.seed, shuffle=True)
        self.x_train = np.reshape(self.x_train.values, (self.x_train.shape[0], self.timesteps, self.data_dim))
        self.x_test = np.reshape(self.x_test.values, (self.x_test.shape[0], self.timesteps, self.data_dim))
        self.y_train = np_utils.to_categorical(self.y_train, num_classes=5)
        self.y_test = np_utils.to_categorical(self.y_test, num_classes=5)


    def model_(self):
        csv_logger = CSVLogger('LSTM_Model_Logger.log')

        tensorflow.random.set_seed(self.seed)
        model = Sequential()
        ##without regularizer
        model.add(LSTM(15, input_shape=(self.timesteps, self.data_dim), return_sequences=True))
        ##with regularizer
        #model.add(LSTM(100, input_shape=(self.timesteps, self.data_dim), return_sequences=True, recurrent_regularizer=regularizers.l2(0.1)))
        #model.add(Dropout(0.1))
        model.add(TimeDistributed(Dense(50)))
        model.add(GlobalAveragePooling1D())
        #model.add(LSTM(50, return_sequences=True, recurrent_regularizer=regularizers.l2(0.1)))
        # model.add(Flatten())
        model.add(Dense(5, activation='softmax'))
        #model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        history = model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                            callbacks=[csv_logger], batch_size=64, epochs=40)
        
        model.save('saved_model/my_model')


        best_val = history.history['val_accuracy'][-1]
        best_acc = history.history['accuracy'][-1]
        best_loss = history.history['loss'][-1]
        best_val_loss = history.history['val_loss'][-1]

        df_ = pd.DataFrame()
        df_.loc[self.seed, 'seed'] = self.seed
        df_.loc[self.seed, 'Timestep'] = self.timesteps
        df_.loc[self.seed, 'Acc'] = best_acc
        df_.loc[self.seed, 'Val_acc'] = best_val
        df_.loc[self.seed, 'Loss'] = best_loss
        df_.loc[self.seed, 'Val_loss'] = best_val_loss

        self.df_best = self.df_best.append(df_)

        #PLOTS
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')

        plt.title('Training and validation accuracy')
        plt.legend()
        plt.show()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')

        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        #CONFUSION MATRIX

        y_pred = model.predict(self.x_test)
        y_test_class = np.argmax(self.y_test, axis=1)
        y_pred_class = np.argmax(y_pred, axis=1)


        #two class
        target_names = ['0', '1']

        # Accuracy of the predicted values
        print(classification_report(y_test_class, y_pred_class, target_names=target_names))
        cm = confusion_matrix(y_test_class, y_pred_class)
        print(cm)

        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax, fmt='g')

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix Base vs Target')
        ax.xaxis.set_ticklabels(['0','1'])
        ax.yaxis.set_ticklabels(['0', '1'])

        plt.show()

    def return_results(self):
        return self.df
