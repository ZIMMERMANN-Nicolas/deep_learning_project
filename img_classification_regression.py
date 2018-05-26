import matplotlib.pyplot as plt
import numpy as np
import keras.utils.np_utils as np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input,  Convolution2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras.optimizers import SGD
import keras
from img_generation import *

def generate_linear_classifier(nb_samples,free_location=False):
    #create model
    model = Sequential()
    nb_neurons = 20
    model.add(Dense(nb_neurons, input_shape=(10000,), activation="relu"))
    model.add(Dense(nb_neurons, input_shape=(10000,), activation="tanh"))
    model.add(Dense(3, activation="softmax"))

    #optimizers & compilation
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    #create training data
    [X_train, Y_train] = generate_dataset_classification(nb_samples, 20,free_location)
    Y_train =keras.utils.to_categorical(Y_train,num_classes=3)

    #training
    model.fit(X_train, Y_train, epochs=20, batch_size=32)

    #print(model.get_weights())

    return model

#This function returns a model classifying moving shapes
def generate_convolutional_network_moving_shapes(nb_samples):
    num_classes = 3
    height =100
    width = 100
    depth = 1
    conv_depth_1 = 15
    kernel_size = 5
    pool_size = 4
    hidden_size = 512
    drop_prob_1 = 0.4
    drop_prob_2 = 0.5
    nb_epochs = 25
    batch_size = 32
    inp = Input(shape=(height, width, depth))

    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    drop_1 = Dropout(drop_prob_1)(pool_1)
    flat = Flatten()(drop_1)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_2 = Dropout(drop_prob_2)(hidden)
    out = Dense(num_classes, activation='softmax')(drop_2)
    model = Model(inputs=inp, outputs=out)
    #create training data
    [X_train, Y_train] = generate_dataset_classification(nb_samples, 20,free_location = True)
    Y_train =keras.utils.to_categorical(Y_train,num_classes=3)
    X_train = np.reshape(X_train, (nb_samples,height,width,depth))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer=sgd, # using the sgd optimizer
              metrics=['accuracy']) # reporting the accuracy

    #training
    model.fit(X_train, Y_train, epochs=nb_epochs, batch_size=batch_size)

    return model

def generate_convolutional_network_predict_triangle(nb_samples):
    vector_size = 6
    height =100
    width = 100
    depth = 1
    conv_depth_1 = 16
    conv_depth_2 = 32
    conv_depth_3 = 64
    conv_depth_4 = 64
    kernel_size = 3
    pool_size = 4
    hidden_size = 512
    drop_prob_1 = 0.2
    drop_prob_2 = 0.1
    drop_prob_3 = 0.1
    drop_prob_4 = 0.1
    nb_epochs = 32
    batch_size = 32
    inp = Input(shape=(height, width, depth))

    conv_1a = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1a)
    norm_1 = BatchNormalization()(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(norm_1)
    drop_1 = Dropout(drop_prob_1)(pool_1)

    conv_2a = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_2 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_2a)
    norm_2 = BatchNormalization()(conv_2)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(norm_2)
    drop_2 = Dropout(drop_prob_2)(pool_2)

    conv_3a = Convolution2D(conv_depth_3, (kernel_size, kernel_size), padding='same', activation='relu')(drop_2)
    conv_3 = Convolution2D(conv_depth_3, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3a)
    norm_3 = BatchNormalization()(conv_3)
    pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(norm_3)
    drop_3 = Dropout(drop_prob_3)(pool_3)

    conv_4a = Convolution2D(conv_depth_4, (kernel_size, kernel_size), padding='same', activation='relu')(drop_3)
    conv_4 = Convolution2D(conv_depth_4, (kernel_size, kernel_size), padding='same', activation='relu')(conv_4a)
    norm_4 = BatchNormalization()(conv_4)
    pool_4 = MaxPooling2D(pool_size=(1, 1))(conv_4)
    drop_4 = Dropout(drop_prob_4)(pool_4)

    flat = Flatten()(drop_4)
    hidden = Dense(hidden_size, activation='relu')(flat)
    hidden_2 = Dense(hidden_size, activation='relu')(hidden)
    out = Dense(vector_size)(hidden_2)
    model = Model(inputs=inp, outputs=out)
    #create training data
    [X_train, Y_train] = generate_dataset_regression(nb_samples)
    X_train_tmp = X_train

    X_train = np.reshape(X_train, (nb_samples,height,width,depth))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', # using the mean_squared_error loss function
                  optimizer=sgd, # using the sgd optimizer
                  metrics=['accuracy']) # reporting the accuracy


    #training
    model.fit(X_train, Y_train, epochs=nb_epochs, batch_size=batch_size)

    return model, X_train_tmp



def prediction(X_test,model,verbose=False):
    X_test = X_test.reshape(1, X_test.shape[0])
    preds = model.predict(X_test)[0]
    max_prov = -1
    ind_prov = -1
    for k in range(len(preds)):
        if(max_prov < preds[k]):
            max_prov = preds[k]
            ind_prov = k
    if verbose:
        if ind_prov == 0:
            print("Rectangle : prob " + str(max_prov))
        elif ind_prov == 1:
            print("Disk : prob " + str(max_prov))
        elif ind_prov == 2:
            print("Triangle : prob " + str(max_prov))
    return ind_prov

def prediction2(X_test,model,verbose=False):
    preds = model.predict(np.array([X_test]))[0]
    max_prov = -1
    ind_prov = -1
    for k in range(len(preds)):
        if(max_prov < preds[k]):
            max_prov = preds[k]
            ind_prov = k
    if verbose:
        if ind_prov == 0:
            print("Rectangle : prob " + str(max_prov))
        elif ind_prov == 1:
            print("Disk : prob " + str(max_prov))
        elif ind_prov == 2:
            print("Triangle : prob " + str(max_prov))
    return ind_prov

def test_model(model,nb_test=100,free_location=False):#a recoder en utilisant generate_test_set_classification()
    X_test = [generate_a_rectangle(free_location=free_location) for k in range(nb_test)]
    good_prediction = 0
    for x in X_test:
        if prediction(x,model) == 0:
            good_prediction += 1
    print("rectangle : ratio " + str(float(good_prediction)/nb_test))
    good_prediction = 0
    X_test = [generate_a_disk(free_location=free_location) for k in range(nb_test)]
    for x in X_test:
        if prediction(x,model) == 1:
            good_prediction += 1
    print("disk : ratio " + str(float(good_prediction)/nb_test))
    good_prediction = 0
    X_test = [generate_a_triangle(free_location=free_location)[0] for k in range(nb_test)]
    for x in X_test:
        if prediction(x[0],model) == 2:#belek
            good_prediction += 1
    print("triangle : ratio " + str(float(good_prediction)/nb_test))

def test_model2(model,nb_test=100,free_location=False):#a recoder en utilisant generate_test_set_classification()
    X_test = [generate_a_rectangle(free_location=free_location) for k in range(nb_test)]
    X_test = np.reshape(X_test, (nb_test,100,100,1))
    good_prediction = 0
    for x in X_test:
        if prediction2(x,model) == 0:
            good_prediction += 1
    print("rectangle : ratio " + str(float(good_prediction)/nb_test))
    good_prediction = 0
    X_test = [generate_a_disk(free_location=free_location) for k in range(nb_test)]
    X_test = np.reshape(X_test, (nb_test,100,100,1))
    for x in X_test:
        if prediction2(x,model) == 1:
            good_prediction += 1
    print("disk : ratio " + str(float(good_prediction)/nb_test))
    good_prediction = 0
    X_test = [generate_a_triangle(free_location=free_location)[0] for k in range(nb_test)]
    X_test = np.reshape(X_test, (nb_test,100,100,1))
    for x in X_test:
        if prediction2(x,model) == 2:
            good_prediction += 1
    print("triangle : ratio " + str(float(good_prediction)/nb_test))


# def main():
#     # generate a linear classifier with a training set size 1000 and figures centered -> work
#     # model = generate_linear_classifier(1000)
#     # test_model(model,100)
#
#     # # generate a linear classifier with a training set size 1000 and figures not centered -> don't work well
#     # model = generate_linear_classifier(1000,True)
#     # test_model(model,100,True)
#
#     # model = generate_convolutional_network_moving_shapes(1000)
#     # test_model2(model,100,True)
#
#
#     model, X_train = generate_convolutional_network_predict_triangle(2)
#
#
#     [X_train_2, Y_train] = generate_dataset_regression(30, 0)
#
#     for x in X_train_2:
#         X_pred = np.reshape(x, (1,100,100,1))
#         Y_pred = model.predict(X_pred)
#         visualize_prediction(x, Y_pred)



    #[X_test, Y_test] = generate_test_set_regression()
