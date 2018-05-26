import matplotlib.pyplot as plt
import numpy as np
import keras.utils.np_utils as np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input,  Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
import keras
import matplotlib.patches as patches

#  This file provides those functions :
#  generate_a_drawing(figsize, U, V, noise=0.0):
#  generate_a_rectangle(noise=0.0, free_location=False):
#  generate_a_disk(noise=0.0, free_location=False):
#  generate_a_triangle(noise=0.0, free_location=False):
#  generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
#  generate_test_set_classification():
#  generate_dataset_regression(nb_samples, noise=0.0):
#  visualize_prediction(x, y):
#  generate_test_set_regression():

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

def generate_test_set_classification(nb, free_location=False, noise=0.0):
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(nb, noise, free_location)
    Y_test = np_utils.to_categorical(Y_test, 3)
    return [X_test, Y_test]


def reorganize_triangle(t):
    i_max = 1
    max_coord = 0
    for i in range(1, 6, 2):
        if(t[i] > max_coord):
            i_max = i
            max_coord = t[i]

    if(i_max != 1):
        t[0], t[1], t[i_max - 1], t[i_max] = t[i_max - 1], t[i_max], t[0], t[1]


    vec1 = [t[2] - t[0], t[3] - t[1], 0]
    vec2 = [t[4] - t[2], t[5] - t[3], 0]
    vec = np.cross(vec1, vec2)
    if(vec[2] < 0):#set anti clockwise
        t[2], t[3], t[4], t[5] = t[4], t[5], t[2], t[3]

    return t

#
# t = [ 1, 0 , 2, 2, 0, 1]
# t = reorganize_triangle(t)
# print(t)

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

    for y in Y:
        y = reorganize_triangle(y)

    return [X, Y]

def visualize_prediction(x, y):
    fig, ax = plt.subplots(figsize=(5, 5))
    I = x.reshape((100,100))
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
