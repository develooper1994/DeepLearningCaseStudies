"""expand_mnist.py
50,000 --> 250,000 images
~~~~~~~~~~~~~~~~~~
Take the 50,000 MNIST training images, and create an expanded set of
250,000 images, by displacing each training image up, down, left and
right, by one pixel.  Save the resulting file to
mnist_expanded.pkl.gz.

Note that this program is memory intensive, and may not run on small
systems.
"""
#### Libraries

# Standard library
import gzip
import os.path
import pickle
import random

import numpy as np

print("Expanding the MNIST training set")

if os.path.exists("mnist_expanded.pkl.gz"):
    print("The expanded training set already exists.  Exiting.")
else:
    # https://github.com/MichalDanielDobrzanski/DeepLearningPython35
    with gzip.open('../../mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    expend_tr_pairs = []
    j = 0
    for x, y in zip(training_data[0], training_data[1]):
        expend_tr_pairs.append((x, y))
        image = np.reshape(x, (-1, 28))
        j += 1
        if j % 1000 == 0:
            print("expending image number", j)
        # iterate over data telling us the details of how to
        # do the displacement
        for d, axis, idx_pos, idx in [
            (1, 0, "first", 0),
            (-1, 0, "first", 27),
            (1, 1, "last", 0),
            (-1, 1, "last", 27)
        ]:
            new_img = np.roll(image, d, axis)  # roll the pixel array
            if idx_pos == "first":
                new_img[idx, :] = np.zeros(28)
            else:
                new_img[:, idx] = np.zeros(28)
            expend_tr_pairs.append((np.reshape(new_img, 28 * 28), y))
    random.shuffle(expend_tr_pairs)
    expend_tr_data = [list(d) for d in zip(*expend_tr_pairs)]
    print("Saving expanded data. This may take a few time. Thanks for your patience")
    # https://github.com/MichalDanielDobrzanski/DeepLearningPython35
    with gzip.open("../../mnist_expanded.pkl.gz", "w") as f:
        pickle.dump((expend_tr_data, validation_data, test_data), f)
    print("Expandation finished")
