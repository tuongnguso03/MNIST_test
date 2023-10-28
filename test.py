def test(model, test_X, test_y):
    test_loss, test_accuracy = model.evaluate(test_X, test_y)
    print('Accuracy on test dataset:', test_accuracy)
    return