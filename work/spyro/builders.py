import tensorflow as tf


def build_actor_critic_mlp(inputs, n_actions, n_layers=3, n_neurons=512, activation="relu",
                           scope_shared="shared", scope_actor="actor", scope_critic="critic"):
    """Build the neural network estimator for actor-critic agents.

    Parameters
    ----------
    inputs: (list of) tf.Placeholder or tf.Tensor
        The inputs for the model. Normally, the (batch of) state observation(s).
    """
    with tf.variable_scope(scope_shared):
        network = build_mlp_body(inputs, n_layers=n_layers, n_neurons=n_neurons,
                                 activation=activation)

    with tf.variable_scope(scope_actor):
        action_probas = add_softmax_layer(network, n_actions)

    with tf.variable_scope(scope_critic):
        value_prediction = add_scalar_regression_layer(network, activation="linear")

    return action_probas, value_prediction


def build_mlp_body(inputs, n_layers=3, n_neurons=512, activation="relu", **kwargs):
    """Build a Multi-Layer Perceptron without a prediction layer."""
    network = tf.contrib.layers.flatten(inputs)
    for l in range(n_layers):
        network = tf.layers.dense(network, n_neurons, activation=activation, **kwargs)
    return network

def add_softmax_layer(inputs, softmax_dim):
    """Add a Softmax output layer to an existing network."""
    output = tf.layers.dense(inputs, softmax_dim, activation=tf.nn.softmax)
    return output


def add_scalar_regression_layer(inputs, activation="linear"):
    output = tf.layers.dense(inputs, 1, activation=activation)
    return output
