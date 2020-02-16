import data_treatment as dt
import keras
import matplotlib.pyplot as plt
import time


class TitanicModel:
    def __init__(self):
        self.model = None

    def build_model(self, input_size=(9, 1), k_reg=keras.regularizers.l2(1e-4), a_reg=keras.regularizers.l1(1e-4)):
        """
        Building a functional keras neural net.
        :param input_size: tuple of input size or input shape respectively
        :param k_reg:
        :param a_reg:
        :return:
        """
        data_input = keras.Input(shape=input_size)
        normed_data = keras.layers.BatchNormalization()(data_input)
        conv1 = keras.layers.Conv1D(7, 1, activation="selu", kernel_regularizer=k_reg, activity_regularizer=a_reg)(normed_data)
        dense1 = keras.layers.Dense(7, activation="selu", kernel_regularizer=k_reg, activity_regularizer=a_reg)(normed_data)
        flat_dense1 = keras.layers.Flatten()(dense1)
        flat_conv1 = keras.layers.Flatten()(conv1)
        concat1 = keras.layers.concatenate([flat_conv1, flat_dense1])
        dense2 = keras.layers.Dense(7, activation="selu", kernel_regularizer=k_reg, activity_regularizer=a_reg)(concat1)
        out = keras.layers.Dense(1, activation="sigmoid")(dense2)
        self.model = keras.Model(inputs=data_input, outputs=out)
        self.model.summary()

    def compiler(self, titanic_optimizer=keras.optimizers.Adam(lr=1e-4), titanic_loss=keras.losses.binary_crossentropy):
        """
        Compiles the model including accuracy metric
        :param titanic_optimizer: keras optimizer, default is set to most common optimizer: Adam optimizer
        :param titanic_loss: keras loss, default is set to most common loss for binary classification: binary_crossentropy
        :return:
        """
        if self.model is None:
            return
        self.model.compile(loss=titanic_loss, optimizer=titanic_optimizer, metrics=["acc"])

    def train(self, train_set=dt.treat_data(dt.load_data("data/train.csv"))):
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
        history = self.model.fit(x=train_data, y=train_labels, epochs=10000, batch_size=100,
                                 callbacks=[my_callback], verbose=2, shuffle=True)
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
    model.build_model((9, 1))
    model.compiler()
    history = model.train()
    plotter(history)
