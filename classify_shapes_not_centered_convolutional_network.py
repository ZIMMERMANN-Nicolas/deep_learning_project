from img_classification_regression import *
from img_generation import *

def main():
    # generate a linear classifier with a training set size 1000 and figures not centered -> don't work well
    model = generate_convolutional_network_moving_shapes(1000)
    test_model2(model,100,True)

main()
