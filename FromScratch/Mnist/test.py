import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_handle()
#  Admit it It is not a worst idea :)
sample_training_data = list(training_data)
sample_training_data = sample_training_data[:1000]
import network, network2
print("-*-*-*-*-*-*-*-*-*network test*-*-*-*-*-*-*-*-*-*-*-")
sizes = [784, 30, 10]
net = network.Network(sizes)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


print("-*-*-*-*-*-*-*-*-*network2 test*-*-*-*-*-*-*-*-*-*-*-")
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
#net.large_weight_initializer()
net.SGD(training_data, 30, 10, 0.1, lmbda = 5.0,evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)


print("-*-*-*-*-*-*-*-*-*Overfitting is treating noise as a signal*-*-*-*-*-*-*-*-*-*-*-")
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(sample_training_data, 400, 10, 0.5, evaluation_data=test_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True)



print("-*-*-*-*-*-*-*-*-*chapter 3 - Regularization (weight decay) example 1\n"
      " (only 1000 of training data and 30 hidden neurons)*-*-*-*-*-*-*-*-*-*-*-")
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(sample_training_data, 400, 10, 0.5,
    evaluation_data=test_data,
    lmbda = 0.1, # this is a regularization parameter
    monitor_evaluation_cost=True,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    monitor_training_accuracy=True)


# chapter 3 - Early stopping implemented
print("-*-*-*-*-*-*-*-*-*chapter 3 - Early stopping implemented*-*-*-*-*-*-*-*-*-*-*-")
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(sample_training_data, 30, 10, 0.5,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_cost=True,
    early_stopping_n=10)


# chapter 4 - The vanishing gradient problem - deep networks are hard to train with simple SGD algorithm
# this network learns much slower than a shallow one.
print("-*-*-*-*-*-*-*-*-*chapter 4 - The vanishing gradient problem - deep networks are hard to train with simple SGD algorithm.\n"
      "this network learns much slower than a shallow one*-*-*-*-*-*-*-*-*-*-*-")
net = network2.Network([784, 30, 30, 30, 30, 10], cost=network2.CrossEntropyCost)
net.SGD(training_data, 30, 10, 0.1,
    lmbda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True)
