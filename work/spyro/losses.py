from __future__ import division
import tensorflow as tf


def quantile_huber_loss(errors, kappa=1, target_axis=1, three_dims=True):
    """Compute the Quantile Huber Loss as defined `Distributional Reinforcement Learning with
    Quantile Regression` (Dabney et al., 2018).

    Parameters
    ----------
    errors: 2-D or 3-D tf.Tensor
        The pairwise errors between target(-quantiles) and predicted quantiles. The first axis
        represents the samples in the batch. If three_dims is True, the second and third axes
        are the target samples and the predicted quantiles in either order (specify this using
        'target_axis'). For a 2D tensor, the second axis has the errors for different quantiles
        for the same observation.
    kappa: int, default=1
        The parameter of Huber Loss, i.e. until what absolute value to use exponential loss.
        By default, kappa=1, which is in line with the paper (Dabney et al., 2018).
    target_axis: int, default=2
        The axis along which the target samples change and the predicted quantiles stay constant.
        See examples below.
    three_dims: bool, default=True
        Whether the input will have three dimensions or two. Three dimensions are used in the
        original paper, but loss may also be calculated over a 2-dimensional tensor in other
        situations.

    Returns
    -------
    qhl_loss: 0-D tf.Tensor (scalar)
        The mean Quantile Huber Loss of the batch as a symbolic tensor.

    Examples
    --------
    >>> # example input when three_dims=True, target_axis=1, and N is the number of atoms
    >>> errors = tf.Tensor([[[Ttheta_1 - theta_1, ..., Ttheta_1 - theta_N],
    >>>                      [Ttheta_2 - theta_1, ..., Ttheta_2 - theta_N],
    >>>                      ...
    >>>                      [Ttheta_N - theta_1, ..., Ttheta_N - theta_N]],
    >>>                     [...same for sample 2 in batch...],
    >>>                     ...
    >>>                     [...same for sample m in batch...]]
    """
    kappa = tf.cast(kappa, tf.float64)
    negative_indicators = tf.cast(errors < 0, tf.float64)
    
    tf_num_atoms = tf.cast(tf.shape(errors)[target_axis], tf.float64)
    taus = tf.range(0.5 * (1 / tf_num_atoms), 1, delta=(1. / tf_num_atoms), dtype=tf.float64, name='tau')
    quantile_weights = tf.abs(tf.subtract(taus, negative_indicators), name="quantile_weights")

    kappa_indicator = tf.cast(tf.abs(errors) < kappa, tf.bool)
    pairwise_huber_loss = tf.where(kappa_indicator,
                          x=tf.multiply(tf.constant(0.5, dtype=tf.float64), tf.cast(tf.square(errors), tf.float64)),
                          y=tf.multiply(tf.constant(0.5, dtype=tf.float64), tf.square(kappa)) + tf.multiply(kappa, (tf.abs(errors) - kappa))
                         )

    # compute Quantile Huber Loss for each pair of Ttheta_j, theta_i
    qhl_pairwise = tf.multiply(quantile_weights, pairwise_huber_loss)

    if three_dims:
        # take expectation over target-samples to obtain E_j[rho(Ttheta_j - theta_i)]
        qhl_for_i = tf.reduce_mean(qhl_pairwise, axis=target_axis, keepdims=False)
    else:
        qhl_for_i = qhl_pairwise

    # take sum over quantiles (there should only be two axes left)
    qhl_per_sample = tf.reduce_sum(qhl_for_i, axis=1, keepdims=False)
    # take average batch loss
    qhl_total = tf.reduce_mean(qhl_per_sample)
    return qhl_total
