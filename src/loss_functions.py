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
    
    # Remove the 1/K since we don't want to favor cases with less objects
#     coward_loss = tf.math.divide_no_nan(
#         tf.reduce_sum(coward_loss_k),
#         1.0
#     )
    
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

            
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, q_min=0.1, reduction=tf.keras.losses.Reduction.AUTO, name="custom_loss"):
        super(CustomLoss, self).__init__(reduction=reduction, name=name)
        self.q_min = q_min
        self.result_values = {
            "attractive": tf.Variable(0.0, trainable=False, name="attractive_loss"),
            "repulsive": tf.Variable(0.0, trainable=False, name="repulsive_loss"),
            "coward": tf.Variable(0.0, trainable=False, name="coward_loss"),
            "noise": tf.Variable(0.0, trainable=False, name="noise_loss"),
            "centroid": tf.Variable(0.0, trainable=False, name="centroid_loss"),
        }

    def call(self, y_true, y_pred):
        object_id = y_true[:, 0]
        true_centroid_x = y_true[:, 1]
        true_centroid_y = y_true[:, 2]
        
        beta = tf.reshape(y_pred[:,:,0:1], [-1, 1])
        x    = tf.reshape(y_pred[:,:,1:3], [-1, 2])
        pred_centroid_x = tf.reshape(y_pred[:,:,3], [-1, 1])
        pred_centroid_y = tf.reshape(y_pred[:,:,4], [-1, 1])
        
        # Compute condensation loss
        loss_dict = condensation_loss(q_min=self.q_min,
                                      object_id=object_id, 
                                      beta=beta,
                                      x=x,
                                      noise_threshold=-1)
        
        # Update result values for the loss components
        self.result_values["attractive"].assign(loss_dict['attractive'])
        self.result_values["repulsive"].assign(loss_dict['repulsive'])
        self.result_values["coward"].assign(loss_dict['coward'])
        self.result_values["noise"].assign(loss_dict['noise'])

        # Calculate `ni` where object_id equals -1
        ni = tf.cast(object_id == -1, tf.float32)

        # Calculate `xi` as (1 - ni) * (atanh(beta) ** 2)
        beta_atanh = tf.math.atanh(beta)
        xi = (1 - ni) * (beta_atanh ** 2)

        # Calculate the regressive loss for centroids (quadrature)
        loss_centroid_x = tf.square(pred_centroid_x - tf.reshape(true_centroid_x, [-1, 1]))
        loss_centroid_y = tf.square(pred_centroid_y - tf.reshape(true_centroid_y, [-1, 1]))
        loss_centroid = tf.sqrt(loss_centroid_x + loss_centroid_y)

        # Weighted sum of centroid losses by xi
        weighted_loss_centroid = tf.reduce_sum(xi * loss_centroid) / tf.reduce_sum(xi)

        # Update result value for centroid loss
        self.result_values["centroid"].assign(weighted_loss_centroid)

        # Combine the losses
        total_loss = (self.result_values["attractive"] +
                      self.result_values["repulsive"] +
                      self.result_values["coward"] +
                      self.result_values["noise"] +
                      self.result_values["centroid"])

        return total_loss

    def get_metrics(self):
        return self.result_values

    def get_config(self):
        config = super(CustomLoss, self).get_config()
        config.update({
            "q_min": self.q_min,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)