from numpy import *
from numpy.core._multiarray_umath import ndarray
import matplotlib.pyplot as plt

'''
This example about single line modelling. y = mx + b (Linear regression)
There is also polinominal models like y = (k=1->inf)sum(mk*x**k) + b =m1*x + m2*x**2 + m3*x**3 + ... + b (Polinom fit)
'''

# y = mx + b
# m is slope, b is y-intercept
def line_regression_result(b, m, x):
    return m * x + b
def toterror(y, x, b, m):
    # SSE(sum of square error)
    return (y - line_regression_result(b, m, x)) ** 2
def toterror_derivative_b(N, y, x, b_current, m_current): return -(2 / N) * (y - line_regression_result(b_current, m_current, x))
def toterror_derivative_m(N, y, x, b_current, m_current): return -(2 / N) * x * (y - line_regression_result(b_current, m_current, x))
def gradient(N, y, x, b_current, m_current):
    return toterror_derivative_b(N, y, x, b_current, m_current), \
           toterror_derivative_m(N, y, x, b_current, m_current)


def error_line_points(b: float, m: float, points):
    total_error: float = 0.0
    n: int = len(points)
    for i in range(n):
        x, y = points[i, (0, 1)]
        total_error += toterror(y, x, b, m)
    return total_error / float(n)


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient: float = 0.0
    m_gradient: float = 0.0
    N: int = len(points)
    for i in range(N):
        x, y = points[i, (0, 1)]
        grad = gradient(N, y, x, b_current, m_current)
        b_gradient += grad[0]
        m_gradient += grad[1]
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b: float = starting_b
    m: float = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return b, m


def run():
    filename = "data.csv"
    points: ndarray = genfromtxt(filename, delimiter=",")
    learning_rate: float = 0.0001
    b_initial: float = 0.0  # initial y-intercept guess
    m_initial: float = 0.0  # initial slope guess
    num_iter: int = 1000
    print(f"Gradient descent started at b = {b_initial}, m = {m_initial}, "
          f"error = {error_line_points(b_initial, m_initial, points)}")
    print("Running 'data.csv'...")

    b, m = gradient_descent_runner(points, b_initial, m_initial, learning_rate, num_iter)
    p = plotter(b, m, points)
    p.title(filename)
    p.show()
    print(f"After ; b = {b}, m = {m}, "
          f"error = {error_line_points(b, m, points)}")

    # another data set but it is random points. Basic line regression isn't good for the random datasets
    points = random.rand(100,2)
    print(f"\nGradient descent started at b = {b_initial}, m = {m_initial}, "
          f"error = {error_line_points(b_initial, m_initial, points)}")
    print("Running 'data.csv'...")

    b, m = gradient_descent_runner(points, b_initial, m_initial, learning_rate, num_iter)
    p = plotter(b, m, points)
    p.title('random data')
    p.show()
    print(f"After ; b = {b}, m = {m}, "
          f"error = {error_line_points(b, m, points)}")


def plotter(b, m, points):
    x, y = points[:, 0], points[:, 1]
    yhat = line_regression_result(b, m, x)

    plt.plot(x, yhat, c='green')
    plt.scatter(x, y, c='red', marker='.', linestyle=':')

    plt.gca().invert_yaxis()
    plt.xlabel('x')
    plt.ylabel('y')
    return plt


if __name__ == '__main__':
    run()
