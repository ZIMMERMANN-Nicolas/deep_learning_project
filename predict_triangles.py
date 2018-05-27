from img_classification_regression import *
from img_generation import *

def main():
    size_training_pool = 2000
    nb_tests = 100
    model, X_train = generate_convolutional_network_predict_triangle(size_training_pool)


    [X, Y] = generate_dataset_regression(nb_tests, 0)

    for x in X:
        X_pred = np.reshape(x, (1,100,100,1))
        Y_pred = model.predict(X_pred)
        visualize_prediction(x, Y_pred)

main()
