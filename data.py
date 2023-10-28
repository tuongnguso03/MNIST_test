from keras.datasets import mnist
import numpy as np

def get_data(dataset = mnist, train_val_ratio = 0.8):
    (train_ds_X, train_ds_y), (test_X, test_y) = dataset.load_data()
    print("Image size: {}x{}".format(train_ds_X.shape[1],train_ds_X.shape[2]))
    #images are uniformly sized, thus no need for resizing.
    data_count = train_ds_X.shape[0]
    train_X = train_ds_X[:int(np.ceil(data_count*train_val_ratio))]
    train_y = train_ds_y[:int(np.ceil(data_count*train_val_ratio))]
    val_X = train_ds_X[int(np.ceil(data_count*train_val_ratio)):]
    val_y = train_ds_y[int(np.ceil(data_count*train_val_ratio)):]
    print("Sample count: Train - {}, Validation - {}, Test - {}".format(train_y.shape[0],val_y.shape[0],test_y.shape[0]))
    return (train_X, train_y), (val_X, val_y), (test_X, test_y)

if __name__ == "__main__":
    print(get_data())