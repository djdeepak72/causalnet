# import numpy as np
# import tensorflow as tf
from tensorflow.keras import regularizers
# import tensorflow.keras.backend as K
from tensorflow.keras.layers import (Concatenate, Dense, Embedding, Input, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_custom_objects

from .metrics import EpsilonLayer, tweedie_loss
from .data_processing import fetch_embedding_weights


def emdedding_net_skeleton(causal_net_data, use_pretrained_model_weights=False, embeddings_path=None, activation='elu'):
    """
    Takes in a custom CausalNetData object and returns inputs and outputs ready for use in a model.
    \nIf you are using weights from a pretrained model, be sure to specify "fixed_weight = True" and the embed_loc filepath.
    """
    
    ######################
    # Fetch causal_net_data components #
    ######################
    get_custom_objects()['tweedie_loss'] = tweedie_loss
    categorical_size_dict = causal_net_data.categorical_size_dict
    categorical_embedding_size = causal_net_data.categorical_embedding_size
    categorical_list = causal_net_data.categorical_variables
    continuous_list = causal_net_data.continuous_variables
    total_categorical_variables = len(categorical_list)
    total_continuous_variables = len(continuous_list)

    if embeddings_path:
        total_variables = total_categorical_variables+total_continuous_variables
        emdedding_weights = fetch_embedding_weights(embeddings_path, categorical_list, total_variables)
    inputs=[]
    concat_layers=[]

    # Create embeddings variable by variable
    for c in categorical_list:
        if use_pretrained_model_weights:
            input_layer = Input(shape=(1,), name=c)
            inputs.append(input_layer)
            embedding_layer = Embedding(input_dim = categorical_size_dict[c],
                                          weights = [emdedding_weights[c]],
                                          output_dim = categorical_embedding_size[c],
                                          input_length = 1,
                                          trainable = False)(input_layer)
        else:
            input_layer = Input(shape=(1,), name=c)    
            inputs.append(input_layer)
            embedding_layer = Embedding(input_dim = categorical_size_dict[c], 
                                          output_dim = categorical_embedding_size[c], 
                                          input_length = 1)(input_layer)

        reshape_layer = Reshape(target_shape = (categorical_embedding_size[c],))(embedding_layer)
        concat_layers.append(reshape_layer)


    # Define the continuous variable layer
    continuous_var_input = Input((total_continuous_variables,), name='cont_vars')
    inputs.append(continuous_var_input)
    
    model_continuous_vars = Dense(128,
                       activation=activation,
                       input_shape = (total_continuous_variables,))(continuous_var_input)

    concat_layers.append(model_continuous_vars)

    concat = Concatenate()(concat_layers)
    
    dense1 = Dense(200,
                  activation=activation)(concat)

    dense2 = Dense(75)(dense1)
    
    dense3 = Dense(1)(dense2)
    print("\n -----------------------------\n Embedding skeleton created!!!")
    return(inputs, dense3)

def dragonnet_with_embeddings(embedding_inputs, embedding_output, reg_l2):
    """
    Neural net predictive model. The dragon has three heads.
    Feed it the embedding model inputs and outputs.
    """
    # t_l1 = 0.
    # t_l2 = reg_l2
    # initializer = tf.keras.initializers.RandomUniform(minval=1e-13, maxval=1.)
    
    get_custom_objects()['tweedie_loss'] = tweedie_loss
    ## representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(embedding_output)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
#     x = Dense(units=200, kernel_initializer='RandomNormal')(embedding_output)
#     x = Dense(units=200, kernel_initializer='RandomNormal')(x)
#     x = Dense(units=200, kernel_initializer='RandomNormal')(x)


    t_predictions = Dense(units=1, activation='sigmoid')(x)

    # HYPOTHESIS
    target0_hidden = Dense(units=100, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_l2))(x)
    target1_hidden = Dense(units=100, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    target0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(target0_hidden)
    target1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(target1_hidden)

    # third
#     y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
#     y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
    target0_prediction = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(target0_hidden)
    target1_prediction = Dense(units=1, activation='sigmoid', kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(target1_hidden)

    eps_layer = EpsilonLayer()
    epsilons = eps_layer(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concatenate_predictions = Concatenate(1)([target0_prediction, target1_prediction, t_predictions, epsilons])
    
    model = Model(inputs=embedding_inputs, outputs=concatenate_predictions)
    
    print("\n -----------------------------\n DragonNet with embeddings initialised!!!.\n")
    return model

def tarnet(input_dim, reg_l2):
    """
    Neural net predictive model. The dragon has three heads.
    :param input_dim:
    :param reg:
    :return:
    """
    get_custom_objects()['tweedie_loss'] = tweedie_loss
    
    inputs = Input(shape=(input_dim,), name='input')

    # representation
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)
    x = Dense(units=200, activation='elu', kernel_initializer='RandomNormal')(x)

    t_predictions = Dense(units=1, activation='sigmoid')(inputs)

    # HYPOTHESIS
    target0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)
    target1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(x)

    # second layer
    target0_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(target0_hidden)
    target1_hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2))(target1_hidden)

    # third
    target0_pred = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(
        target0_hidden)
    target1_pred = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(
        target1_hidden)

    eps_layer = EpsilonLayer()
    epsilons = eps_layer(t_predictions, name='epsilon')
    # logging.info(epsilons)
    concat_pred = Concatenate(1)([target0_pred, target1_pred, t_predictions, epsilons])
    model = Model(inputs=inputs, outputs=concat_pred)

    return model
