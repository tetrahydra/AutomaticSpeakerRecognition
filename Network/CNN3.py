from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam_v2


class CNN3:

    def CNNModel3(self, input_size, class_count):

        model = Sequential()

        model.add(Conv2D(64, (3,3), activation='relu',
                         input_shape=(128, 128, 1)))
        model.add(Conv2D(64, (3,3), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(Conv2D(128, (3,3), activation='relu'))
        model.add(MaxPooling2D(2, 2))

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
