from img_classification_regression import *
from img_generation import *

def main():
    # generate a linear classifier with a training set size 1000 and figures not centered -> don't work well
    model = generate_linear_classifier(1000,True)
    test_model(model,100,True)

main()
