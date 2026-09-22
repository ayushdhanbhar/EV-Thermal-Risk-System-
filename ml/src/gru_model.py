from keras import Sequential
from keras.layers import Dense, Dropout, GRU


def build_gru_model(input_shape, num_classes, units=(64, 32), dropout_rate=0.2):
    model = Sequential([
        GRU(units[0], return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        GRU(units[1]),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
