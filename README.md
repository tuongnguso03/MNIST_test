# MNIST_test
## Installing requirements
`pip install tensorflow`\
`pip install keras-tuner`
## Run code
Run the tuning, training and testing:\
`python convNN.py <path_to_model>` \
(if `<path_to_model>` left empty, save to default)\
\
Run test on the saved model:\
`python convNN.py test <path_to_model>`\
By default, models are saved in `'./checkpoints//model.keras'`
## Results
By accident, I only saved the parameters of the model :P
| Hyperparameter    | Best value |
| -------- | ------- |
| conv_1_filters  | 32    |
| conv_2_filters | 24     |
| dense_1    | 192    |
| dropout_rate    | 0.2    | \

Accuracy on test dataset: 0.9909999966621399

