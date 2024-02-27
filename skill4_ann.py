import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
class DeepANN():
    def simple_model(self, input_shape=(28, 28, 3), optimizer='sgd'):
        model = Sequential()
        model.add(Flatten())
        model.add(Flatten(input_shape=(124, 124)))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model

    def train_model(self, model_inst, train_gen, val_gen, epochs=10):
        mhist = model_inst.fit(train_gen, validation_data=val_gen, epochs=epochs)
        return mhist

    def compare_models(self, models, train_gen, val_gen, epochs=10):
        hist = []
        for model in models:
            history = self.train_model(model, train_gen, val_gen, epochs=epochs)
            hist.append(history)

        plt.figure(figsize=(10, 8))
        for i, history in enumerate(hist):
            plt.plot(history.history['accuracy'], label=f'Model {i + 1}')

        plt.title('Model Training Accuracy Comparison')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show(block=True)
