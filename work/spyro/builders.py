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
        action_probas = tf.layers.dense(inputs, n_actions, activation=tf.nn.softmax)

    with tf.variable_scope(scope_critic):
        value_prediction = tf.layers.dense(inputs, 1, activation="linear")

    return action_probas, value_prediction


def build_mlp_body(inputs, n_layers=3, n_neurons=512, activation="relu", **kwargs):
    """Build a Multi-Layer Perceptron without a prediction layer."""
    network = tf.contrib.layers.flatten(inputs)
    for l in range(n_layers):
        network = tf.layers.dense(network, n_neurons, activation=activation, **kwargs)
    return network


def build_mlp_regressor(inputs, n_layers, n_neurons, activation="relu", output_dim=1, **kwargs):
    network = build_mlp_body(inputs, n_layers=n_layers, n_neurons=n_neurons,
                             activation=activation, **kwargs)
    output = tf.layers.dense(inputs, output_dim, activation="linear")
    return output


def build_dqn(inputs, n_actions, n_layers=2, n_neurons=512, activation="relu", dueling=True,
              value_neurons=64, advantage_neurons=256, **kwargs):
    """Create a (Dueling) Deep Q-Network. Inputs should normally be the states and outputs
    are the predicted values of each action from these states.
    """
    network = build_mlp_body(inputs, n_layers=n_layers, n_neurons=n_neurons,
                             activation=activation, **kwargs)
    if dueling:
        # value stream
        value_layer = tf.layers.dense(network, value_neurons, activation="relu", name="value_layer")
        value = tf.layers.dense(value_layer, 1, activation="linear", name="state_value")
        # advantage stream
        advantage_layer = tf.layers.dense(network, advantage_neurons, activation="relu", name="advantage_layer")
        advantage = tf.layers.dense(advantage_layer, n_actions, activation="linear", name="advantage")
        # combine
        qvalues = value - tf.subtract(advantage, tf.reduce_mean(advantage)  )

    else:
        qvalues = tf.layers.dense(network, n_actions, activation="linear")

    return qvalues
