# Pulled from (https://github.com/object-condensation/object_condensation/blob/main/src/object_condensation/tensorflow/losses.py)
# Needed to change the `tf.math.maximum` 0 -> 0.0 to fix an error from Tensorflow
import tensorflow as tf
def condensation_loss(
    *,
    q_min: float,
    object_id: tf.Tensor,
    beta: tf.Tensor,
    x: tf.Tensor,
    weights: tf.Tensor = None,
    noise_threshold: int = 0,
):
    """Condensation losses

    Args:
        beta: Condensation likelihoods
        x: Clustering coordinates
        object_id: Labels for objects. Objects with `object_id <= 0` are considered
            noise
        weights: Weights per hit, multiplied to attractive/repulsive potentials
        q_min: Minimal charge
        noise_threshold: Threshold for noise hits. Hits with ``object_id <= noise_thld``
            are considered to be noise

    Returns:
        Dictionary of scalar tensors.

        ``attractive``: Averaged over object, then averaged over all objects.
        ``repulsive``: Averaged like ``attractive``
        ``coward``: Averaged over all objects
        ``noise``: Averaged over all noise hits
    """
    if weights is None:
        weights = tf.ones_like(beta)
    q_min = tf.cast(q_min, tf.float32)
    object_id = tf.reshape(object_id, (-1,))
    beta = tf.cast(beta, tf.float32)
    x = tf.cast(x, tf.float32)
    weights = tf.cast(weights, tf.float32)
    not_noise = object_id > noise_threshold
    unique_oids, _ = tf.unique(object_id[not_noise])
    q = tf.cast(tf.math.atanh(beta) ** 2 + q_min, tf.float32)
    mask_att = tf.cast(object_id[:, None] == unique_oids[None, :], tf.float32)
    mask_rep = tf.cast(object_id[:, None] != unique_oids[None, :], tf.float32)
    alphas = tf.argmax(beta * mask_att, axis=0)
    beta_k = tf.gather(beta, alphas)
    q_k = tf.gather(q, alphas)
    x_k = tf.gather(x, alphas)

    dist_j_k = tf.norm(x[None, :, :] - x_k[:, None, :], axis=-1)

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


class LossComponentsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Extract model outputs and y_true from logs if needed
        output = self.model.predict(self.validation_data[0], batch_size=batch_size)
        y_true = self.validation_data[1]

        # Calculate loss components
        beta = tf.reshape(output[:,:,0:1],[-1,1])
        x    = tf.reshape(output[:,:,1:3],[-1,2])
        loss_dict = condensation_loss(q_min=0.1,
                                      object_id=y_true, 
                                      beta=beta,
                                      x=x,
                                      noise_threshold=0)

        logs['attractive_loss'] = loss_dict['attractive']
        logs['repulsive_loss'] = loss_dict['repulsive']
        logs['coward_loss'] = loss_dict['coward']
        if 'noise' in loss_dict:
            logs['noise_loss'] = loss_dict['noise']

        # Print the loss components
        print(f"Epoch {epoch + 1}:")
        print(f" - Attractive Loss: {logs['attractive_loss']:.4f}")
        print(f" - Repulsive Loss: {logs['repulsive_loss']:.4f}")
        print(f" - Coward Loss: {logs['coward_loss']:.4f}")
        if 'noise_loss' in logs:
            print(f" - Noise Loss: {logs['noise_loss']:.4f}")
            

# Custom loss function
def custom_loss(object_id, output):
    beta = tf.reshape(output[:,:,0:1],[-1,1])
    x    = tf.reshape(output[:,:,1:3],[-1,2])
    loss_dict = condensation_loss(q_min=0.1,
                                  object_id=object_id, 
                                  beta=beta,
                                  x=x,
                                  noise_threshold=0)

    return loss_dict['attractive'] + loss_dict['repulsive'] + loss_dict['coward'] # + loss_dict['noise']

# Custom metric functions
def attractive_loss_metric(y_true, y_pred):
    # Calculate attractive loss only
    beta = tf.reshape(y_pred[:,:,0:1],[-1,1])
    x    = tf.reshape(y_pred[:,:,1:3],[-1,2])
    loss_dict = condensation_loss(q_min=0.1,
                                  object_id=y_true, 
                                  beta=beta,
                                  x=x,
                                  noise_threshold=0)
    return loss_dict['attractive']

def repulsive_loss_metric(y_true, y_pred):
    # Calculate repulsive loss only
    beta = tf.reshape(y_pred[:,:,0:1],[-1,1])
    x    = tf.reshape(y_pred[:,:,1:3],[-1,2])
    loss_dict = condensation_loss(q_min=0.1,
                                  object_id=y_true, 
                                  beta=beta,
                                  x=x,
                                  noise_threshold=0)
    return loss_dict['repulsive']

def coward_loss_metric(y_true, y_pred):
    # Calculate coward loss only
    beta = tf.reshape(y_pred[:,:,0:1],[-1,1])
    x    = tf.reshape(y_pred[:,:,1:3],[-1,2])
    loss_dict = condensation_loss(q_min=0.1,
                                  object_id=y_true, 
                                  beta=beta,
                                  x=x,
                                  noise_threshold=0)
    return loss_dict['coward']

def noise_loss_metric(y_true, y_pred):
    # Calculate noise loss only
    beta = tf.reshape(y_pred[:,:,0:1],[-1,1])
    x    = tf.reshape(y_pred[:,:,1:3],[-1,2])
    loss_dict = condensation_loss(q_min=0.1,
                                  object_id=y_true, 
                                  beta=beta,
                                  x=x,
                                  noise_threshold=0)
    return loss_dict.get('noise', 0.0)