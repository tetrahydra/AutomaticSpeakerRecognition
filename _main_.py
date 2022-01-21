try:
    import numpy as np
    import numpy as np
    import sys
    import os
    from sklearn.model_selection import train_test_split
except:
    print("Error importing global modules")

try:
    from Dataset.Dataset import Dataset
    from Network.CNN import CNN
    from Network.Functions import Functions
except:
    print("Error importing local modules")


class SpeakerRecognition(Dataset, CNN, Functions):
    def __init__(self) -> None:
        pass

    def SaveToFile(self, X_data, Y_data):
        np.save('X.npy', X_data)
        np.save('Y.npy', Y_data)

    def LoadFile(self):
        X_data = np.load('X.npy')
        Y_data = np.load('Y.npy')
        return X_data, Y_data

    @staticmethod
    def ScaleMinMax(X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def SplitInput(self, X, Y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.3,
                                                            random_state=123)
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    SimoNN_App = SpeakerRecognition()

    folder = "/Users/halim/RUC/DATASET (SimoNN) Voice/Voice-cleaned"

    number_of_volunteers = 6

    '''
    Argument to prepare dataset
    '''
    if sys.argv[1] == "dataset":

        print("Preparing dataset")
        X, Y = SimoNN_App.PrepareDataset(folder)
        print(X.shape)

        print("Saving to numpy file")
        SimoNN_App.SaveToFile(X, Y)

    '''
    Argument to start the training process
    '''
    if sys.argv[1] == "training":
        print("Loading dataset")
        X_data, Y_data = SimoNN_App.LoadFile()

        # X_data = SimoNN_App.ScaleMinMax(X_data)

        print("Randomizing and splitting dataset")
        X_train, X_test, y_train, y_test = SimoNN_App.SplitInput(
            X_data, Y_data)

        print("Expanding the dimensions from a vector to a 2D")
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        # X_train = X_train.reshape(X_train.shape[0], 128, 128)
        # X_test = X_test.reshape(X_test.shape[0], 128, 128)

        print("Building model")
        model = SimoNN_App.BuildModel(
            input_size=X_train.shape[1], class_count=number_of_volunteers)

        print("Training model")
        
        history = SimoNN_App.Training(model, X_train, y_train, X_test, y_test)

        print("Evaluating model")
        score, acc = SimoNN_App.Evaluate(model, X_test, y_test, batch_size=16)

        print("Saving the plot")
        SimoNN_App.SavePlot(history)

        print("Save model")
        SimoNN_App.SaveModel(model)

    '''
    Argument to load a trained model and make a prediction using original data
    '''
    if sys.argv[1] == "predict1":

        X_data, Y_data = SimoNN_App.LoadFile()

        X_data = SimoNN_App.ScaleMinMax(X_data)

        # For experiment 3 and 4
        # X_data = X_data.reshape(X_data.shape[0], 128, 128)
        
        print("Load model")
        model_loaded = SimoNN_App.LoadModel()

        print("Predicting using new data")
        predictions = model_loaded.predict(X_data)

        parent_folder = os.path.abspath("/Users/halim/simonn-framework")
        sys.path.insert(0, parent_folder)

        from SimoNN.Metrics.Confusion_Matrix import Confusion_Matrix

        output_cm = Confusion_Matrix(number_of_volunteers, Y_data, predictions)

        list_volunteer_index = [91, 200, 27, 9, 84, 72]

        output_cm.Summary(precede="Volunteer", labels=list_volunteer_index)

        output_cm.SummaryF2(precede="Volunteer", labels=list_volunteer_index)

        output_cm.Plot(axis_label=list_volunteer_index)
    
    '''
    Argument to load a trained model and make a prediction using original data
    '''
    if sys.argv[1] == "predict2":

        X_data, Y_data = SimoNN_App.LoadFile()

        X_data = SimoNN_App.ScaleMinMax(X_data)

        # For experiment 3 and 4
        X_data = X_data.reshape(X_data.shape[0], 128, 128)
        
        print("Load model")
        model_loaded = SimoNN_App.LoadModel()

        print("Predicting using new data")
        predictions = model_loaded.predict(X_data)

        parent_folder = os.path.abspath("/Users/halim/simonn-framework")
        sys.path.insert(0, parent_folder)

        from SimoNN.Metrics.Confusion_Matrix import Confusion_Matrix

        output_cm = Confusion_Matrix(number_of_volunteers, Y_data, predictions)

        list_volunteer_index = [91, 200, 27, 9, 84, 72]

        output_cm.Summary(precede="Volunteer", labels=list_volunteer_index)

        output_cm.SummaryF2(precede="Volunteer", labels=list_volunteer_index)

        output_cm.Plot(axis_label=list_volunteer_index)
