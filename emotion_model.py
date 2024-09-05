from keras.models import model_from_json
from keras.optimizers import SGD

class EmotionModel:
    """
    Class for loading, compiling, and predicting emotions using a neural network model.

    Attributes:
    model: The neural network model that has been loaded and compiled.

    Methods:
    __init__(self, architecture_path, weights_path):
        Initializes the class by loading and compiling the model.

    load_model(self, architecture_path, weights_path):
        Loads the model from the provided architecture and weights files.

    compile_model(self):
        Compiles the model with the SGD optimizer and 'categorical_crossentropy' loss function.

    predict(self, face_features):
        Makes a prediction of emotion based on the provided facial features.
    """

    def __init__(self, architecture_path, weights_path):
        """
        Initializes the class by loading and compiling the model.

        Parameters:
        architecture_path (str): Path to the JSON file containing the model architecture.
        weights_path (str): Path to the file containing the model weights.
        """
        self.model = self.load_model(architecture_path, weights_path)
        self.compile_model()
        
    def load_model(self, architecture_path, weights_path):
        """
        Loads the model from the provided architecture and weights files.

        Parameters:
        architecture_path (str): Path to the JSON file containing the model architecture.
        weights_path (str): Path to the file containing the model weights.

        Returns:
        keras.Model: The loaded Keras model.
        """
        with open(architecture_path, 'r') as file:
            model = model_from_json(file.read())
        model.load_weights(weights_path)
        return model
    
    def compile_model(self):
        """
        Compiles the model with the SGD optimizer and 'categorical_crossentropy' loss function.
        """
        sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    def predict(self, face_features):
        """
        Makes a prediction of emotion based on the provided facial features.

        Parameters:
        face_features (numpy.ndarray): The facial features to predict emotions for.

        Returns:
        numpy.ndarray: The model's prediction for the given facial features.
        """
        return self.model.predict(face_features.reshape(1, 48, 48, 1))
