from img_classification_regression import *
from img_generation import *

def main():
    # generate a linear classifier with a training set size 1000 and figures not centered -> don't work well
    # model = generate_convolutional_network_moving_shapes(10)
    # test_model2(model,10,True)
    nb_tests = 100
    size_training_pool = 1000

    # generate a linear classifier with a training set size 1000 and figures centered -> work
    model = generate_convolutional_network_moving_shapes(size_training_pool)
    [X_test, Y_test] = generate_test_set_classification(nb_tests, True, 0.2)#Test model on 100 shapes
    X_test = np.reshape(X_test, (nb_tests,100,100,1))
    nb_correct_predictions = 0

    # X_test = X_test.reshape(1, X_test.shape[0])
    # preds = model.predict(X_test)[0]

    for k in range(len(X_test)):
        pred = model.predict(np.array([X_test[k]]))
        i_pred = get_max_index(pred[0])
        i_real = get_max_index(Y_test[k])
        if(i_pred == i_real):
            nb_correct_predictions += 1

    print("Result prediction : "
          + str(nb_correct_predictions)
          + " correct predictions over "
          + str(len(X_test))
          + " shapes predicted.")
    print("Ratio: " + str(float(nb_correct_predictions) / len(X_test)))

def get_max_index(array):
    index = 0
    max_index = 0
    for i in range(len(array)):
        if(array[i] > max_index):
            max_index = array[index]
            index = i

    return index

main()
