# emotion_model.py
from keras.models import model_from_json
from keras.optimizers import SGD

class EmotionModel:
    def __init__(self, architecture_path, weights_path):
        self.model = self.load_model(architecture_path, weights_path)
        self.compile_model()
        
    def load_model(self, architecture_path, weights_path):
        with open(architecture_path, 'r') as file:
            model = model_from_json(file.read())
        model.load_weights(weights_path)
        return model
    
    def compile_model(self):
        sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)
    
    def predict(self, face_features):
        return self.model.predict(face_features.reshape(1, 48, 48, 1))
