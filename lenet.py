# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Reshape
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


class LeNet:
    @staticmethod
    def build(numChannels, width, height, numClasses,
              activation="relu", weightsPath=None):
        # initialize the model
        model = Sequential()
        inputShape = (width, height, numChannels)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, width, height)

        # define the first set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(32, 5, padding="same",
                         input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the second set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(64, 3, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the third set of CONV => ACTIVATION => POOL layers
        model.add(Conv2D(128, 3, padding="same"))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # define the first FC => ACTIVATION layers
        # model.add(Reshape(target_shape=(16*16*64, ))) # 2 sets, 64x64 photos
        # model.add(Reshape(target_shape=(8*8*64, ))) # 2 sets, 32x32 photos
        # model.add(Reshape(target_shape=(8*8*128, ))) # dla 3 zbiorów CONV => ACTIVATION => POOL, 64x64
        model.add(Reshape(target_shape=(4*4*128, ))) # dla 3 zbiorów CONV => ACTIVATION => POOL, 32x32
        # model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))

        # define the second FC layer
        model.add(Dense(numClasses))

        # lastly, define the soft-max classifier
        model.add(Activation("softmax"))

        # if a weights path is supplied (inicating that the model was
        # pre-trained), then load the weights
        if weightsPath is not None:
            model.load_weights(weightsPath)

        # return the constructed network architecture
        return model