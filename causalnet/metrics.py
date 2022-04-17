import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import binary_accuracy

# Define variable categories and embedding sizes before converting data to numpy arrays for modeling

def custom_objects():
    '''
    custom_objects functions for when loading the model
    '''
    custom_objects = {'EpsilonLayer':EpsilonLayer, 
                 'tweedie_loss':tweedie_loss,
                 'binary_classification_loss':binary_classification_loss,
                 'dragonnet_tweedieloss_binarycross':dragonnet_tweedieloss_binarycross,
                 'treatment_accuracy':treatment_accuracy,
                 'track_epsilon':track_epsilon,
                 'make_tarreg_loss':targeted_regularisation,
                 'tarreg_ATE_unbounded_domain_loss':tarreg_ATE_unbounded_domain_loss}
    return custom_objects

class EpsilonLayer(Layer):
    '''
    '''

    def __init__(self, units=1, **kwargs):
        super().__init__(**kwargs)
        self.units = units        
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.epsilon = self.add_weight(name='epsilon',
                                       shape=[1, self.units],
                                       initializer='RandomNormal',
                                       trainable=True)
        super().build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.epsilon * tf.ones_like(inputs)[:, 0:1]
    
    def get_config(self):
        config = super().get_config()
        return {**config, 'units':self.units}

def binary_classification_loss(concat_true, concat_pred):
    treatment_actual = concat_true[:, 1]
    treatment_prediction = concat_pred[:, 2]
    treatment_prediction = (treatment_prediction + 0.001) / 1.002
    bce = tf.keras.losses.BinaryCrossentropy()
    treatment_loss = tf.reduce_sum(bce(treatment_actual, treatment_prediction))

    return treatment_loss

def rmse(target_actual, target_prediction):
    return K.sqrt(K.mean(K.square(target_prediction - target_actual)))

def regression_loss(concat_true, concat_pred):
    target_actual = concat_true[:, 0]
    treatment_actual = concat_true[:, 1]

    target0_prediction = concat_pred[:, 0]
    target1_prediction = concat_pred[:, 1]

    regression_loss0 = tf.reduce_sum((1. - treatment_actual) * tf.square(target_actual - target0_prediction))
    regression_loss1 = tf.reduce_sum(treatment_actual * tf.square(target_actual - target1_prediction))

    return regression_loss0 + regression_loss1

def custom_relu(x):
    return(K.greater(1e-9,x))

def leaky_relu(x):
    return(K.greater(0.01*x, x))

def tweedie_loss(concat_true, concat_pred):
    # From https://towardsdatascience.com/tweedie-loss-function-for-right-skewed-data-2c5ca470678f
    p=1.75
    
    if concat_true.shape[1] > 1:
        target_actual = concat_true[:, 0]
        treatment_actual = concat_true[:, 1]
        
        target0_prediction = concat_pred[:, 0]
        target1_prediction = concat_pred[:, 1]
    else:
        treatment_actual = 1.
        target_actual = concat_true
        target0_prediction = 0.00001
        target1_prediction = concat_pred
 
    
    target_actual = tf.clip_by_value(target_actual, clip_value_min =1e-10, clip_value_max=999)
    target0_prediction = tf.clip_by_value(target0_prediction, clip_value_min=1e-10, clip_value_max=999)
    target1_prediction = tf.clip_by_value(target1_prediction, clip_value_min=1e-10, clip_value_max=999)
    
    tweedie_loss0 =  tf.reduce_mean(1. - treatment_actual) * tf.reduce_mean(
                                            -target_actual * tf.pow(target0_prediction, 1-p)/(1-p) +\
                                            tf.pow(target0_prediction, 2-p)/(2-p)
                                                )
                             
    
    tweedie_loss1 = tf.reduce_mean(treatment_actual) * tf.reduce_mean(
                                            -target_actual * tf.pow(target1_prediction, 1-p)/(1-p) +\
                                            tf.pow(target1_prediction, 2-p)/(2-p)
                                                )
                             

    return tweedie_loss0 + tweedie_loss1

def ned_loss(concat_true, concat_pred):
    treatment_actual = concat_true[:, 1]

    treatment_prediction = concat_pred[:, 1]
    return tf.reduce_sum(K.binary_crossentropy(treatment_actual, treatment_prediction))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)

def dragonnet_tweedieloss_binarycross(concat_true, concat_pred):
    return tweedie_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)

def treatment_accuracy(concat_true, concat_pred):
    treatment_actual = concat_true[:, 1]
    treatment_prediction = concat_pred[:, 2]
    return binary_accuracy(treatment_actual, treatment_prediction)

def track_epsilon(concat_true, concat_pred):
    epsilons = concat_pred[:, 3]
    return tf.abs(tf.reduce_mean(epsilons))

def targeted_regularisation(ratio=1., dragonnet_loss=dragonnet_tweedieloss_binarycross):
    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        target_actual = concat_true[:, 0]
        treatment_actual = concat_true[:, 1]

        target0_prediction = concat_pred[:, 0]
        target1_prediction = concat_pred[:, 1]
        treatment_prediction = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        treatment_prediction = (treatment_prediction + 0.01) / 1.02

        target_prediction = treatment_actual * target1_prediction + (1 - treatment_actual) * target0_prediction

        h = treatment_actual / treatment_prediction - (1 - treatment_actual) / (1 - treatment_prediction)

        y_pert = target_prediction + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(target_actual - y_pert))

        loss = vanilla_loss + ratio * targeted_regularization
        return loss

    return tarreg_ATE_unbounded_domain_loss

def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla_loss = dragonnet_loss(concat_true, concat_pred)

        target_actual = concat_true[:, 0]
        treatment_actual = concat_true[:, 1]

        target0_prediction = concat_pred[:, 0]
        target1_prediction = concat_pred[:, 1]
        treatment_prediction = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        treatment_prediction = (treatment_prediction + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        target_prediction = treatment_actual * target1_prediction + (1 - treatment_actual) * target0_prediction

        h = treatment_actual / treatment_prediction - (1 - treatment_actual) / (1 - treatment_prediction)

        y_pert = target_prediction + epsilons * h
        targeted_regularization = tf.reduce_sum(tf.square(target_actual - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        return loss
