# Pulled from (https://github.com/object-condensation/object_condensation/blob/main/src/object_condensation/tensorflow/losses.py)
# Needed to change the `tf.math.maximum` 0 -> 0.0 to fix an error from Tensorflow
import tensorflow as tf
import os
def check_for_nans(tensor, message):
    tf.debugging.check_numerics(tensor, message)

# Added 9/4/2024
# Gradient of tf.norm() has the sum**2 in the denominator
# This may cause overflow if sum**2 is near 0
# Define safe norm function to counter this
def safe_norm(x, epsilon=1e-12, axis=None):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)


def condensation_loss(
    *,
    q_min: float,
    object_id: tf.Tensor,
    event_id: tf.Tensor = None,
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
    if event_id is None:
        event_id = tf.ones_like(object_id)
    else:
        event_id  = tf.reshape(event_id,(-1,))
        
    beta = tf.cast(beta, tf.float32)
    x = tf.cast(x, tf.float32)
    weights = tf.cast(weights, tf.float32)
    
    not_noise = object_id > noise_threshold
    unique_oids, _ = tf.unique(object_id[not_noise])
    
    q = tf.cast(tf.math.atanh(beta) ** 2 + q_min, tf.float32)
    
    # Adjust the masks to ensure compatible shapes
    mask_att = tf.cast((object_id[:, None] == unique_oids[None, :]), tf.float32)
    mask_rep = tf.cast((object_id[:, None] != unique_oids[None, :]), tf.float32) 

    # Get sorted indices of object_id
    sorted_indices = tf.argsort(object_id)
    
    # Sort object_id and event_id according to sorted indices
    sorted_object_id = tf.gather(object_id, sorted_indices)
    sorted_event_id = tf.gather(event_id, sorted_indices)
    
    # Find first occurrence of each unique_oid
    first_occurrence_indices = tf.searchsorted(sorted_object_id, unique_oids)
    
    # Gather the corresponding event_id for the first occurrence of each unique_oid
    unique_event_ids = tf.gather(sorted_event_id, first_occurrence_indices)
    
    mask_evt = tf.cast(event_id[:,None]   == unique_event_ids[None,:], tf.float32)
    mask_rep = mask_rep * mask_evt
    
    alphas = tf.argmax(beta * mask_att, axis=0)
    beta_k = tf.gather(beta, alphas)
    q_k = tf.gather(q, alphas)
    x_k = tf.gather(x, alphas)
    
    dist_j_k = safe_norm(x[None, :, :] - x_k[:, None, :], axis=-1)
    
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
    
    

    v_att = tf.math.divide_no_nan(
        tf.reduce_sum(v_att_k), tf.cast(tf.shape(unique_oids)[0], tf.float32)
    )

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
    
    v_rep = tf.math.divide_no_nan(
        tf.reduce_sum(v_rep_k), tf.cast(tf.shape(unique_oids)[0], tf.float32)
    )

    coward_loss_k = 1.0 - beta_k
    coward_loss = tf.math.divide_no_nan(
        tf.reduce_sum(coward_loss_k),
        tf.cast(tf.shape(unique_oids)[0], tf.float32),
    )
    
    noise_loss = tf.math.divide_no_nan(
        tf.reduce_sum(beta[object_id <= noise_threshold]),
        tf.reduce_sum(tf.cast(object_id <= noise_threshold, tf.float32)),
    )

    return {
        "attractive": v_att,
        "repulsive": v_rep,
        "coward": coward_loss,
        "noise": noise_loss,
    }

            
def calculate_losses(y_true, y_pred, q_min, batch_size, K, single_event_loss=True):
    object_id = tf.cast(y_true[:, :, 0], tf.int32)  # (batch_size, 100, 1)
    beta      = y_pred[:, :, 0:1]                   # (batch_size, 100, 1)
    x         = y_pred[:, :, 1:3]                   # (batch_size, 100, 2)
    if single_event_loss==True:
        event_id = tf.tile(tf.range(batch_size)[:, tf.newaxis], [1, K])
    else:
        event_id = None
    object_id = tf.reshape(object_id,[-1])
    beta      = tf.reshape(beta,[-1,1])
    x         = tf.reshape(x,[-1,2])
    loss_dict = condensation_loss(q_min=q_min,
                                   object_id=object_id,
                                   event_id=event_id,
                                   beta=beta,
                                   x=x,
                                   noise_threshold=-1)

    return loss_dict

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, q_min=0.1, reduction=tf.keras.losses.Reduction.SUM, name="custom_loss", single_event_loss=True):
        super(CustomLoss, self).__init__(reduction=reduction, name=name)
        self.q_min = q_min
        self.single_event_loss = single_event_loss

    def call(self, y_true, y_pred):
            batch_size = tf.shape(y_true)[0] 
            K          = tf.shape(y_true)[1]
            loss_dict = calculate_losses(y_true, y_pred, self.q_min, batch_size, K, self.single_event_loss)
    
    
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
        batch_size = tf.shape(y_true)[0] 
        K          = tf.shape(y_true)[1]
        loss_dict = calculate_losses(y_true, y_pred, self.q_min, batch_size, K)
        return super(AttractiveLossMetric, self).update_state(loss_dict['attractive'], sample_weight)

class RepulsiveLossMetric(tf.keras.metrics.Mean):
    def __init__(self, q_min=0.1, name="repulsive_loss", **kwargs):
        super(RepulsiveLossMetric, self).__init__(name=name, **kwargs)
        self.q_min = q_min

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0] 
        K          = tf.shape(y_true)[1]
        loss_dict = calculate_losses(y_true, y_pred, self.q_min, batch_size, K)
        return super(RepulsiveLossMetric, self).update_state(loss_dict['repulsive'], sample_weight)

class CowardLossMetric(tf.keras.metrics.Mean):
    def __init__(self, q_min=0.1, name="coward_loss", **kwargs):
        super(CowardLossMetric, self).__init__(name=name, **kwargs)
        self.q_min = q_min

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0] 
        K          = tf.shape(y_true)[1]
        loss_dict = calculate_losses(y_true, y_pred, self.q_min, batch_size, K)
        return super(CowardLossMetric, self).update_state(loss_dict['coward'], sample_weight)

class NoiseLossMetric(tf.keras.metrics.Mean):
    def __init__(self, q_min=0.1, name="noise_loss", **kwargs):
        super(NoiseLossMetric, self).__init__(name=name, **kwargs)
        self.q_min = q_min

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0] 
        K          = tf.shape(y_true)[1]
        loss_dict = calculate_losses(y_true, y_pred, self.q_min, batch_size, K)
        return super(NoiseLossMetric, self).update_state(loss_dict['noise'], sample_weight)