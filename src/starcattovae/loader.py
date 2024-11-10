class Loader:
    def __init__(self, model_path):
        self.model = Autoencoder.load(model_path)

    def predict(self, data):
        return self.model.predict(data)