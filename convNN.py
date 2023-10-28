from data import get_data
from train import tune_and_train
from test import test
from tensorflow import keras
from keras.datasets import mnist
import tensorflow as tf



def main():
    #get data
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = get_data(mnist)
    #train and save model
    model = tune_and_train(train_X, train_y, val_X, val_y)
    model.save_weights('./checkpoints/my_checkpoint')
    #run test
    test(model, test_X, test_y)

if __name__ == "__main__":
    main()