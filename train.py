import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

def model_builder(hp):
    model = keras.Sequential()
    #Convolution layer 1
    hp_Conv_1_filters = hp.Int('conv_1_filters', min_value=8, max_value=32, step=8)
    model.add(keras.layers.Conv2D(hp_Conv_1_filters, 3, padding = "same", activation='relu', input_shape=(28, 28, 1)),)
    model.add(keras.layers.MaxPooling2D((2, 2), strides=2))

    #Convolution layer 2
    hp_Conv_2_filters = hp.Int('conv_2_filters', min_value=8, max_value=32, step=8)
    model.add(keras.layers.Conv2D(hp_Conv_2_filters, 3, padding = "same", activation='relu', input_shape=(28, 28, 1)),)
    model.add(keras.layers.MaxPooling2D((2, 2), strides=2))

    #Dropout rate
    hp_dropout_rate = hp.Choice('learning_rate', values=[0.1, 0.15, 0.2, 0.25])

    model.add(keras.layers.Flatten())
    
    #Dense layer 1
    model.add(keras.layers.Dropout(hp_dropout_rate),)
    hp_dense_1 = hp.Int('dense_1', min_value=32, max_value=256, step=32)
    model.add(keras.layers.Dense(units=hp_dense_1, activation='relu'))
    
    #Output layer
    model.add(keras.layers.Dropout(hp_dropout_rate),)
    model.add(keras.layers.Dense(10))

    model.compile(optimizer='adam',
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def tune_and_train(train_X, train_y, val_X, val_y):
    tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='./',
                     project_name='MNIST_test')
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(train_X, train_y, epochs=50, validation_data=(val_X, val_y), callbacks=[stop_early])
    #Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    model.fit(train_X, train_y, epochs = 40, batch_size = 100, validation_data=(val_X, val_y))
    return model
