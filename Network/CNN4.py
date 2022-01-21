from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import adam_v2


class CNN4:

    def CNNModel4(self, input_size, class_count):

        model = Sequential()

        model.add(Conv1D(64, 3, activation='relu',
                         input_shape=(input_size, 1)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(256, 3, activation='relu'))
        model.add(Conv1D(256, 3, activation='relu'))
        model.add(MaxPooling1D(3))

        model.add(Flatten())

        model.add(Dropout(0.5))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(class_count, activation='softmax'))

        lr = 0.0007
        epochs = 30

        optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
