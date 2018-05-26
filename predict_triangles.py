from img_classification_regression import *
from img_generation import *

def main():
    model, X_train = generate_convolutional_network_predict_triangle(20)


    [X_train_2, Y_train] = generate_dataset_regression(30, 0)

    for x in X_train_2:
        X_pred = np.reshape(x, (1,100,100,1))
        Y_pred = model.predict(X_pred)
        visualize_prediction(x, Y_pred)

main()
