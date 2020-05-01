def and_gate(x1, x2):
    # z = input * weight + bias
    # a = activate(z)
    # to make it simple I will give the weights

    # and gate table look like this (in numpy)...
    # and_gate_table =   ([
    #                     [np.array([[np.array([0, 0, 0]).reshape(1, -1), 1],
    #                     [np.array([0,0,0]).reshape(1, -1), 0],
    #                     [np.array([0,1,0]).reshape(1, -1), 0],
    #                     [np.array([0,1,0]).reshape(1, -1), 0],
    #                     [np.array([1,1,1]).reshape(1, -1), 1],
    #                     ])

    w0 = -0.2
    # weights should be equal and positive for and gate
    w1 = 0.5
    w2 = 0.5
    z = w0 + w1 * x1 + w2 * x2
    # I have made my activation function
    activation = lambda x: 1 if x >= 0.5 else 0
    a = activation(z)
    print(a)

# test the table
and_gate(0, 0)
and_gate(0, 1)
and_gate(1, 0)
and_gate(1, 1)
