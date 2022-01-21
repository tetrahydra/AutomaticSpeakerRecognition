from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import adam_v2

class CNN1:

    def CNNModel1(self, input_size, class_count):

        model = Sequential()
        model.add(Conv1D(32, 3, activation='relu',
                         input_shape=(input_size, 1)))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D())

        model.add(Conv1D(512, 3, activation='relu'))
        model.add(MaxPooling1D())
        
        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))

        #model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.5))

        model.add(Dense(class_count, activation='softmax'))

        lr = 0.001
        epochs = 30

        optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)

        model.compile(optimizer='adam',
                      #learning_rate=0.0007,
                      loss='mean_squared_error',
                      metrics=['accuracy'])

        return model
