from data import get_data
from train import tune_and_train
from test import test
from tensorflow import keras
from keras.datasets import mnist
import tensorflow as tf
import sys



def main(model_path = './checkpoints/model.keras'):
    #get data
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = get_data(mnist)
    #train and save model
    model = tune_and_train(train_X, train_y, val_X, val_y)
    model.save(model_path)
    #run test
    test(model, test_X, test_y)

if __name__ == "__main__":
    match len(sys.argv)-1:
        case 0:
            main()
        case 1:
            if sys.argv[1][-6:] == ".keras"
                main(sys.argv[1])
            else: 
                raise TypeError("Incorrect file path")    
        case 2:
            if sys.argv[1] == 'test':
                (train_X, train_y), (val_X, val_y), (test_X, test_y) = get_data(mnist)
                model = new_model = tf.keras.models.load_model(sys.argv[2])
                test(model, test_X, test_y)
            else: 
                raise TypeError("Incorrect arguments")