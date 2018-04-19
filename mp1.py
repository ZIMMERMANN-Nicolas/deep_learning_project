import matplotlib.pyplot as plt
import numpy as np
import keras.utils.np_utils as np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input,  Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
import keras

def generate_a_drawing(figsize, U, V, noise=0.0):
    fig = plt.figure(figsize=(figsize,figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    plt.close(fig)
    return imdata

def generate_a_rectangle(noise=0.0, free_location=False):
    figsize = 1.0
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing(figsize, U, V, noise)


def generate_a_disk(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_triangle(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(figsize, U, V, noise)
    return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]


# im = generate_a_rectangle(10, True)
# plt.imshow(im.reshape(100,100), cmap='gray')
# im = generate_a_disk(10)
# plt.imshow(im.reshape(100,100), cmap='gray')
# [im, v] = generate_a_triangle(20, False)
# plt.imshow(im.reshape(100,100), cmap='gray')

def generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros(nb_samples)
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1:
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

# dataset = generate_dataset_classification(10, noise=0.0, free_location=True)
# for d in dataset[0]:
#     plt.imshow(d.reshape(100,100), cmap='gray')
#     plt.show()

def generate_test_set_classification():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(3, 20, True)#300
    Y_test = np_utils.to_categorical(Y_test, 3)
    return [X_test, Y_test]

def generate_linear_classifier(nb_samples):
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
    [X_train, Y_train] = generate_dataset_classification(nb_samples, 20)
    Y_train =keras.utils.to_categorical(Y_train,num_classes=3)

    #training
    model.fit(X_train, Y_train, epochs=20, batch_size=32)

    #print(model.get_weights())

    return model


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

def test_model(model,nb_test=100):
    X_test = [generate_a_rectangle() for k in range(nb_test)]
    good_prediction = 0
    for x in X_test:
        if prediction(x,model) == 0:
            good_prediction += 1
    print("rectangle : ratio " + str(float(good_prediction)/nb_test))
    good_prediction = 0
    X_test = [generate_a_disk() for k in range(nb_test)]
    for x in X_test:
        if prediction(x,model) == 1:
            good_prediction += 1
    print("disk : ratio " + str(float(good_prediction)/nb_test))
    good_prediction = 0
    X_test = [generate_a_triangle()[:-1] for k in range(nb_test)]
    for x in X_test:
        if prediction(x[0],model) == 2:
            good_prediction += 1
    print("triangle : ratio " + str(float(good_prediction)/nb_test))


model = generate_linear_classifier(2000)
test_model(model,10)


def generate_dataset_regression(nb_samples, noise=0.0):
    # Getting im_size:
    im_size = generate_a_triangle()[0].shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros([nb_samples, 6])
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i)
        [X[i], Y[i]] = generate_a_triangle(noise, True)
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]


import matplotlib.patches as patches

def visualize_prediction(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((72,72))
    ax.imshow(I, extent=[-0.15,1.15,-0.15,1.15],cmap='gray')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])

    xy = y.reshape(3,2)
    tri = patches.Polygon(xy, closed=True, fill = False, edgecolor = 'r', linewidth = 5, alpha = 0.5)
    ax.add_patch(tri)

    plt.show()

def generate_test_set_regression():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_regression(300, 20)
    Y_test = np_utils.to_categorical(Y_test, 3)
    return [X_test, Y_test]
