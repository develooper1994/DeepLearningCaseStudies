""" This dummy method is a naive way to classify from total black pixel numbers and values.
how many correct answer ?
Not much if i compare to neural networks.
"""
# libraries
# S.L.
from collections import defaultdict

import mnist_loader


def main():
    """how many correct answer ?"""
    training_data, validation_data, test_data = mnist_loader.load_data()

    avgs = avg_darknesses(training_data)

    test = [int(guess_digit(image, avgs) == digit)
            for image, digit in zip(test_data[0], test_data[1])]  # how many correct answer ?
    num_correct = sum(test)

    print("Baseline classifier using average darkness of image.")
    print("{0} of {1} values correct.".format(num_correct, len(test_data[1])))


def avg_darknesses(training_data):
    digit_counts = defaultdict(int)
    darknesses = defaultdict(float)
    # count all pixels and sum all the pixel values for specific digit.
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    avgs = defaultdict(float)
    # calculate the average value for all digits.
    for digit, n in digit_counts.items():
        avgs[digit] = darknesses[digit] / n
    return avgs


def guess_digit(image, avgs):
    darkness = sum(image)
    distances = {k: abs(v - darkness) for k, v in avgs.items()}
    return min(distances, key=distances.get)


if __name__ == "__main__":
    main()
