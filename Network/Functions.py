from matplotlib import pyplot as plt
import json
from tensorflow.keras.models import save_model, load_model


class Functions:

    def BuildModel(self, input_size, class_count):
        model = self.CNNModel2(input_size, class_count)
        model.summary()
        return model

    def Training(self, model, X_train, y_train, X_validation, y_validation):
        history = model.fit(X_train,
                            y_train,
                            batch_size=64,
                            epochs=20,
                            validation_data=(X_validation, y_validation))
        # Save the weights
        model.save_weights('./checkpoints/')
        return history

    def Evaluate(self, model, X_test, y_test, batch_size):
        score, acc = model.evaluate(X_test, y_test, batch_size)
        return score, acc

    def SavePlot(self, history):
        with open('history.json', 'w') as file:
            json.dump(history.history, file)

        plt.plot(history.history['loss'])
        plt.plot(history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('foo.png', bbox_inches='tight')

    def SaveModel(self, model):
        filepath = './saved_model'
        save_model(model, filepath)

    def LoadModel(self):
        filepath = './saved_model'
        return load_model(filepath)