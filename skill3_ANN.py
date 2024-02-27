from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
class DeepANN():
    def simple_model(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Flatten(input_shape=(124, 124)))
        model.add(Dense(128,activation="relu"))
        model.add(Dense(64,activation = "relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model