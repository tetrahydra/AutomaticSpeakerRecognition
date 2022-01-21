from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.optimizers import adam_v2


class CNN2:

    def CNNModel2(self, input_size, class_count):

        model = Sequential()

        model.add(Conv1D(64, 3, activation='relu',
                         input_shape=(input_size, 1)))
        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(3))

        model.add(Conv1D(128, 3, activation='relu'))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(3))

        model.add(Flatten())

        model.add(Dropout(0.4))

        model.add(Dense(256, activation='relu'))

        model.add(Dropout(0.3))

        model.add(Dense(class_count, activation='softmax'))

        lr = 0.0007
        epochs = 20

        optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)

        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model
