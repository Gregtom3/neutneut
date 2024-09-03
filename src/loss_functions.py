# Pulled from (https://github.com/object-condensation/object_condensation/blob/main/src/object_condensation/tensorflow/losses.py)
# Needed to change the `tf.math.maximum` 0 -> 0.0 to fix an error from Tensorflow
import tensorflow as tf

def check_for_nans(tensor, message):
    tf.debugging.check_numerics(tensor, message)

def condensation_loss(
    *,
    q_min: float,
    object_id: tf.Tensor,
    beta: tf.Tensor,
    x: tf.Tensor,
    weights: tf.Tensor = None,
    noise_threshold: int = 0,
):
    """Condensation losses with NaN checks."""

    if weights is None:
        weights = tf.ones_like(beta)
    
    q_min = tf.cast(q_min, tf.float32)
    object_id = tf.reshape(object_id, (-1,))
    beta = tf.cast(beta, tf.float32)
    x = tf.cast(x, tf.float32)
    weights = tf.cast(weights, tf.float32)
    
    check_for_nans(beta, "NaN detected after beta casting and clipping")
    check_for_nans(x, "NaN detected after x casting")
    
    not_noise = object_id > noise_threshold
    unique_oids, _ = tf.unique(object_id[not_noise])
    
    q = tf.cast(tf.math.atanh(beta) ** 2 + q_min, tf.float32)
    check_for_nans(q, "NaN detected after calculating q")
    
    mask_att = tf.cast(object_id[:, None] == unique_oids[None, :], tf.float32)
    mask_rep = tf.cast(object_id[:, None] != unique_oids[None, :], tf.float32)
    
    alphas = tf.argmax(beta * mask_att, axis=0)
    beta_k = tf.gather(beta, alphas)
    q_k = tf.gather(q, alphas)
    x_k = tf.gather(x, alphas)
    
    check_for_nans(beta_k, "NaN detected after gathering beta_k")
    check_for_nans(q_k, "NaN detected after gathering q_k")
    check_for_nans(x_k, "NaN detected after gathering x_k")
    
    dist_j_k = tf.norm(x[None, :, :] - x_k[:, None, :], axis=-1)
    check_for_nans(dist_j_k, "NaN detected after calculating dist_j_k")

    v_att_k = tf.math.divide_no_nan(
        tf.reduce_sum(
            q_k
            * tf.transpose(weights)
            * tf.transpose(q)
            * tf.transpose(mask_att)
            * dist_j_k**2,
            axis=1,
        ),
        tf.reduce_sum(mask_att, axis=0) + 1e-9,
    )
    check_for_nans(v_att_k, "NaN detected after calculating v_att_k")

    v_att = tf.math.divide_no_nan(
        tf.reduce_sum(v_att_k), tf.cast(tf.shape(unique_oids)[0], tf.float32)
    )
    check_for_nans(v_att, "NaN detected after calculating v_att")
    
    v_rep_k = tf.math.divide_no_nan(
        tf.reduce_sum(
            q_k
            * tf.transpose(weights)
            * tf.transpose(q)
            * tf.transpose(mask_rep)
            * tf.math.maximum(0.0, 1.0 - dist_j_k),
            axis=1,
        ),
        tf.reduce_sum(mask_rep, axis=0) + 1e-9,
    )
    check_for_nans(v_rep_k, "NaN detected after calculating v_rep_k")

    v_rep = tf.math.divide_no_nan(
        tf.reduce_sum(v_rep_k), tf.cast(tf.shape(unique_oids)[0], tf.float32)
    )
    check_for_nans(v_rep, "NaN detected after calculating v_rep")

    coward_loss_k = 1.0 - beta_k
    coward_loss = tf.math.divide_no_nan(
        tf.reduce_sum(coward_loss_k),
        tf.cast(tf.shape(unique_oids)[0], tf.float32),
    )
    
    check_for_nans(coward_loss, "NaN detected after calculating coward_loss")

    noise_loss = tf.math.divide_no_nan(
        tf.reduce_sum(beta[object_id <= noise_threshold]),
        tf.reduce_sum(tf.cast(object_id <= noise_threshold, tf.float32)),
    )
    check_for_nans(noise_loss, "NaN detected after calculating noise_loss")

    return {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": coward_loss,
        "noise": noise_loss,
    }

            
def calculate_losses(y_true, y_pred, q_min):
    object_id = tf.cast(y_true[:, :, 0], tf.int32)

    beta = tf.reshape(y_pred[:, :, 0:1], [-1, 1])
    x = tf.reshape(y_pred[:, :, 1:3], [-1, 2])

    # Compute condensation loss
    loss_dict = condensation_loss(q_min=q_min,
                                  object_id=object_id,
                                  beta=beta,
                                  x=x,
                                  noise_threshold=-1)

    return loss_dict


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, q_min=0.1, reduction=tf.keras.losses.Reduction.SUM, name="custom_loss"):
        super(CustomLoss, self).__init__(reduction=reduction, name=name)
        self.q_min = q_min

    def call(self, y_true, y_pred):
        loss_dict = calculate_losses(y_true, y_pred, self.q_min)

        # Combine the losses
        total_loss = (loss_dict['attractive'] +
                      loss_dict['repulsive'] +
                      loss_dict['coward'] +
                      loss_dict['noise'])

        return total_loss

    def get_config(self):
        config = super(CustomLoss, self).get_config()
        config.update({"q_min": self.q_min})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
class AttractiveLossMetric(tf.keras.metrics.Mean):
    def __init__(self, q_min=0.1, name="attractive_loss", **kwargs):
        super(AttractiveLossMetric, self).__init__(name=name, **kwargs)
        self.q_min = q_min

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_dict = calculate_losses(y_true, y_pred, self.q_min)
        return super(AttractiveLossMetric, self).update_state(loss_dict['attractive'], sample_weight)

class RepulsiveLossMetric(tf.keras.metrics.Mean):
    def __init__(self, q_min=0.1, name="repulsive_loss", **kwargs):
        super(RepulsiveLossMetric, self).__init__(name=name, **kwargs)
        self.q_min = q_min

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_dict = calculate_losses(y_true, y_pred, self.q_min)
        return super(RepulsiveLossMetric, self).update_state(loss_dict['repulsive'], sample_weight)

class CowardLossMetric(tf.keras.metrics.Mean):
    def __init__(self, q_min=0.1, name="coward_loss", **kwargs):
        super(CowardLossMetric, self).__init__(name=name, **kwargs)
        self.q_min = q_min

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_dict = calculate_losses(y_true, y_pred, self.q_min)
        return super(CowardLossMetric, self).update_state(loss_dict['coward'], sample_weight)

class NoiseLossMetric(tf.keras.metrics.Mean):
    def __init__(self, q_min=0.1, name="noise_loss", **kwargs):
        super(NoiseLossMetric, self).__init__(name=name, **kwargs)
        self.q_min = q_min

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_dict  = calculate_losses(y_true, y_pred, self.q_min)
        return super(NoiseLossMetric, self).update_state(loss_dict['noise'], sample_weight)