import data_treatment as dt
import keras
import matplotlib.pyplot as plt
import time


class TitanicModel:
    def __init__(self):
        self.model = None

    def build_model(self, input_size=(7,)):
        """
        Building a functional keras neural net.
        :param input_size: tuple of input size or input shape respectively
        :return:
        """
        data_input = keras.Input(shape=input_size)
        normed_data = keras.layers.BatchNormalization()(data_input)
        conv1 = keras.layers.Conv1D(7, 1)(normed_data)
        dense1 = keras.layers.Dense(7, activation="relu")(normed_data)
        flat_conv1 = keras.layers.Flatten()(conv1)
        concat1 = keras.layers.concatenate([flat_conv1, dense1])
        out = keras.layers.Dense(1, activation="softmax")(concat1)
        self.model = keras.Model(inputs=data_input, outputs=out)
        self.model.summary()

    def compiler(self, titanic_optimizer=keras.optimizers.Adam, titanic_loss=keras.losses.binary_crossentropy):
        """
        Compiles the model including accuracy metric
        :param titanic_optimizer: keras optimizer, default is set to most common optimizer: Adam optimizer
        :param titanic_loss: keras loss, default is set to most common loss for binary classification: binary_crossentropy
        :return:
        """
        if self.model is None:
            return
        self.model.compile(loss=titanic_loss, optimizer=titanic_optimizer, metrics=["acc"])

    def train(self, train_set=dt.treated_data(dt.load_data("data/train.csv")), test_set=dt.treated_data(dt.load_data("data/test.csv"))):
        """
        Trains the model with early stopping. Train set and test set are converted from strings to floats and normalized
        in order to avoid values greater than 1
        :return:
        """
        if self.model is None:
            return
        my_callback = keras.callbacks.callbacks.EarlyStopping(monitor='loss', min_delta=0.0, patience=1000, verbose=2,
                                                              mode='auto', baseline=None, restore_best_weights=False)

        train_data, train_labels = train_set
        test_data, test_labels = test_set
        history = self.model.fit(x=train_data, y=train_labels, validation_data=(test_data, test_labels), epochs=10000,
                                 callbacks=[my_callback], verbose=2, shuffle=True, batch_size=100)
        return history


def plotter(history):
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("titanic_loss_" + str(time.time()) + ".png")

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("titanic_acc_" + str(time.time()) + ".png")


if __name__ == "__main__":
    model = TitanicModel()
    model.build_model((7,))
    model.compiler()
    history = model.train()
    plotter(history)
