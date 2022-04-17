import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from IPython.display import clear_output
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TerminateOnNaN)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import get_custom_objects

from .metrics import *
from .metrics import rmse, tweedie_loss
from .data_processing import *
from .model import *


class TrainPerformanceViz(tf.keras.callbacks.Callback):
    '''
    One of the callbacks which plots training performance metrics like loss & accuracy while training the model
    TODO: Can be replaced by Tensorboard in the future
    '''
    
    # Called when training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.val_losses = []
        self.logs = []
    
    # Called at the end of every epoch to collect logs
    def on_epoch_end(self, epoch, logs={}):
        
        # Get  logs, losses and accuracy info from logs
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        # Check if more 1 epochs have passed
        if len(self.losses) > 1:
            
            # clear previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            plt.style.use("seaborn-bright")
            
            # Setting up plots for accuracy & loss for training & validation sets by epochs
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()
            
def output_process(prediction, treatment, target, target_scaler, x_data, idx, weight):
    try:
        q_t0 = target_scaler.inverse_transform(prediction[:, 0].copy())/weight
        q_t1 = target_scaler.inverse_transform(prediction[:, 1].copy())/weight
    except:
        q_t0 = target_scaler.inverse_transform(prediction[:, 0].reshape(-1,1).copy())/weight
        q_t1 = target_scaler.inverse_transform(prediction[:, 1].reshape(-1,1).copy())/weight
        
    try:
        g = prediction[:, 2].copy()
    except:
        g = prediction[:, 2].reshape(-1,1).copy()

    if prediction.shape[1] == 4:
        eps = prediction[:, 3][0]
    else:
        eps = np.zeros_like(prediction[:, 2])

    target = target_scaler.inverse_transform(target.copy())/weight
    var = "Average propensity(prediction) for treated: {} and untreated: {}".format(g[treatment.squeeze() == 1.].mean(),
                                                                        g[treatment.squeeze() == 0.].mean())
    print(var)

    return {'q_t0': q_t0, 'q_t1': q_t1, 'g': g, 't': treatment, 'y': target, 'x': x_data, 'weight': weight, 'index': idx, 'eps': eps}

def embeddings_train(causal_net_data, 
                    optimizer='adam',
                    loss='tweedie_loss',
                    epochs=50,
                    batch_size=128,
                    validation_split = .3,
                    activation = 'sigmoid',
                    metrics = ['mean_squared_logarithmic_error','tweedie_loss'],
                    callbacks = [
                        TrainPerformanceViz(),
                        TerminateOnNaN(),
                        EarlyStopping(monitor='tweedie_loss', mode='min', patience=5, min_delta=0, restore_best_weights = True),
                        ReduceLROnPlateau(monitor='tweedie_loss', mode='min', factor=0.5, patience=0, 
                                         min_delta=1e-15, cooldown=2, min_lr=1e-15),
                        ModelCheckpoint(
                            filepath=Path('embed_checkpoint')/'model.{epoch:02d}-{loss:.2f}', monitor='tweedie_loss', mode='min', verbose=0, save_best_only=False,
                            save_weights_only=False, save_freq='epoch')

                                ]):

    get_custom_objects()['tweedie_loss'] = tweedie_loss
    
        
    print('Splitting the  causal_net_data object.')
    target_data      = causal_net_data.target_preprocess
    weight_data = causal_net_data.weight_preprocess
    training_index = causal_net_data.training_index
    test_index  = causal_net_data.test_index   

    x_train = {}
    x_test = {}
    for key in causal_net_data.x_preprocess:
        x_train[key] = causal_net_data.x_preprocess[key][training_index]
        x_test[key] = causal_net_data.x_preprocess[key][test_index]

    weight_train = weight_data[training_index]

    y_train = target_data[training_index]
    
    try:
        print("Checking for saved checkpoints.")
        list_of_folders = glob.glob('embed_checkpoint'+'/*')
        latest_file = max(list_of_folders, key=os.path.getctime)
        print("Checkpoint found & loaded from " + latest_file)
        embedding_model = load_model(latest_file)
    except:
        print("Checkpoint not found. Initialising embedding model to train.")
        inputs, outputs = emdedding_net_skeleton(causal_net_data, activation=activation)
        embedding_model = Model(inputs, outputs)

    embedding_model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics,
              loss_weights = weight_train)
        
    embedding_model.fit(x = x_train, 
                y = y_train,
                epochs = epochs,
                callbacks = callbacks,
                batch_size = batch_size,
                validation_split = validation_split
                 )
    return embedding_model
    
def dragonnet_first_stage(causal_net_data,
                   checkpoint_path,
                   embeddings_path=None,
                   targetted_regularisation=True,                                       
                   knob_loss=dragonnet_loss_binarycross, 
                   ratio=1.,
                   dragonnet_model=None,
                   val_split = 0.2,
                   rd_one_batch=15,
                   rd_one_epoch=100,
                   verbose=0,
                   adam_lr=.001,
                   adam_b1=.9,
                   adam_b2=.999,
                   adam_eps=1e-01,
                   adam_amsgrad=True):
    
    parent_checkpoint_path = Path(checkpoint_path)
    adam_checkpt = parent_checkpoint_path/'adam_checkpoints'
    get_custom_objects()['tweedie_loss'] = tweedie_loss
    
    ########################
    ## Unpack Dragon Data ##
    ########################
    print("Splitting causal_net_data object")
    treatment_data      = causal_net_data.treatment_preprocess
    target_data      = causal_net_data.target_preprocess
    training_index = causal_net_data.training_index
    test_index  = causal_net_data.test_index

    x_train = {}
    x_test = {}
    for key in causal_net_data.x_preprocess:
        x_train[key] = causal_net_data.x_preprocess[key][training_index]
        x_test[key] = causal_net_data.x_preprocess[key][test_index]
            
    treatment_train = treatment_data[training_index]
    target_train = target_data[training_index]        
    target_treat_train = np.concatenate([target_train, treatment_train],1)
    
        
    metrics = [tweedie_loss, treatment_accuracy, track_epsilon,binary_classification_loss]

    if targetted_regularisation:
        loss = targeted_regularisation(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss
        
    import time
    start_time = time.time()
    
    print("\n---------------------------------------------------------\n Starting to train DragonNet : Stage 1 !!!!!!\n")
    

    ##If you want to continue to train the dragonnet model from a checkpoint, set the dragonnet argument to that model
    if dragonnet_model:
        dragonnet = dragonnet_model
    else:
        if embeddings_path:
            print('Using embeddings specified in the `embeddings_path` argument')
            inputs, outputs = emdedding_net_skeleton(causal_net_data, use_pretrained_model_weights=True, embeddings_path=embeddings_path)
        else:
            inputs, outputs = emdedding_net_skeleton(causal_net_data)
        dragonnet = dragonnet_with_embeddings(inputs, outputs, 0.01)
        
    dragonnet.compile(
        optimizer=Adam(learning_rate=adam_lr, 
                 beta_1=adam_b1, 
                 beta_2=adam_b2, 
                 epsilon=adam_eps,
                 amsgrad=adam_amsgrad),
        loss=loss, 
        metrics=metrics)
        
    adam_callbacks = [
        TrainPerformanceViz(),
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', mode='min', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0),
        ModelCheckpoint(
            filepath=adam_checkpt/'model.{epoch:02d}-{val_loss:.2f}', monitor='val_loss', mode='min', verbose=0, save_best_only=False,
            save_weights_only=False, save_freq='epoch')]
    dragonnet.fit(x_train, 
                  target_treat_train,
                  callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=rd_one_epoch,
                  batch_size=rd_one_batch, 
                  verbose=verbose)

    elapsed_time = time.time() - start_time
    print("Stage 1 has run for ", elapsed_time, " seconds")
    
    return(dragonnet)

def dragonnet_second_stage(dragonnet_first_stage_model,
                   causal_net_data,
                   checkpoint_path,
                   targetted_regularisation=True,                                       
                   knob_loss=dragonnet_loss_binarycross, 
                   ratio=1., 
                   dragon='dragonnet',
                   validation_split = 0.2,
                   rd_two_batch=64,
                   rd_two_epoch=300,
                   verbose=0,
                   sgd_lr=1e-5,
                   sgd_momentum=0.99,
                   sgd_nesterov=True):
    
    checkpoint_parent_path = Path(checkpoint_path)
    sgd_checkpoint = checkpoint_parent_path/'sgd_checkpoints/'
    get_custom_objects()['tweedie_loss'] = tweedie_loss
    
    ########################
    ## Split causal_net_data ##
    ########################
    print("Unpacking the dragon data object")
    treatment_data      = causal_net_data.treatment_preprocess
    target_data      = causal_net_data.target_preprocess
    target_scaler    = causal_net_data.target_scaler
    weight_data = causal_net_data.weight_preprocess
    training_idx = causal_net_data.training_index
    test_idx  = causal_net_data.test_index

    x_train = {}
    x_test = {}
    for item in causal_net_data.x_preprocess:
        x_train[item] = causal_net_data.x_preprocess[item][training_idx]
        x_test[item] = causal_net_data.x_preprocess[item][test_idx]
            
    treatment_train, treatment_test = treatment_data[training_idx], treatment_data[test_idx]
    weight_train, weight_test = weight_data[training_idx], weight_data[test_idx]
    
    target_train, target_test = target_data[training_idx], target_data[test_idx]    
        
    target_treat_train = np.concatenate([target_train, treatment_train],1)
    
    metrics = [ binary_classification_loss,tweedie_loss, treatment_accuracy, track_epsilon]

    if targetted_regularisation:
        loss = targeted_regularisation(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss
        
    import time
    start_time = time.time()
    print("\n---------------------------------------------------------\n Starting to train DragonNet : Stage 2 !!!!!!!!\n")
    
    sig_optimizer = SGD(learning_rate=sgd_lr, momentum=sgd_momentum, nesterov=sgd_nesterov)

    dragonnet_first_stage_model.compile(optimizer=sig_optimizer,
                      loss=loss, 
                      metrics=metrics)
    
    sgd_callbacks = [
        TrainPerformanceViz(),
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0),
        ModelCheckpoint(
            filepath=sgd_checkpoint/'model.{epoch:02d}-{val_loss:.2f}', monitor='val_loss', mode='min', verbose=0, save_best_only=False,
            save_weights_only=False, save_freq='epoch')
                    ]

    dragonnet_first_stage_model.fit(x_train, 
                  target_treat_train, 
                  callbacks=sgd_callbacks,
                  validation_split=validation_split,
                  epochs=rd_two_epoch,
                  batch_size=rd_two_batch,
                  verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** Stage 2 has run for ", elapsed_time, " seconds")

    target_treat_prediction_test = dragonnet_first_stage_model.predict(x_test)
    target_treat_prediction_train = dragonnet_first_stage_model.predict(x_train)

    train_result = []
    test_results = []
    
    test_results += [output_process(target_treat_prediction_test, treatment_test, target_test, target_scaler, x_test, test_idx, weight_test)]
    train_result += [output_process(target_treat_prediction_train, treatment_train, target_train, target_scaler, x_train, training_idx, weight_train)]
    K.clear_session()

    return test_results, train_result, dragonnet_first_stage_model

    

def train_dragonnet_w_embeddings(causal_net_Data,
                           checkpoint_path,
                           embeddings_path=None,
                           targeted_regularization=True,                                       
                           knob_loss=dragonnet_loss_binarycross, 
                           ratio=1., 
                           val_split = 0.2,
                           rd_one_batch=64,
                           rd_one_epoch=100,
                           rd_two_batch=64,
                           rd_two_epoch=100,
                           verbose=0,
                           adam_lr=.001,
                           adam_b1=.9,
                           adam_b2=.999,
                           adam_eps=1e-01,
                           adam_amsgrad=True,
                           sgd_lr=1e-5,
                           sgd_momentum=0.99,
                           sgd_nesterov=True):
    """
    Takes inputs and creates embeddings for categorical variables and feeds it into the dragonnet. The combination of the categorical and continuous lists should be all the covariates you want to include in your model.
    
    \nRequired formats:
    \ndragon_data: DragonData object
    \ncheckpoint_loc: where do you want checkpointed models to be saved  
    """
    
    checkpoint_parent = Path(checkpoint_path)
    get_custom_objects()['tweedie_loss'] = tweedie_loss
    
    ########################
    ## Unpack Dragon Data ##
    ########################

    treatment_data      = causal_net_Data.treatment_preprocess
    target_data      = causal_net_Data.target_preprocess
    target_scaler    = causal_net_Data.target_scaler
    weight_data = causal_net_Data.weight_preprocess
    training_idx = causal_net_Data.training_index
    test_idx  = causal_net_Data.test_index   

    x_train = {}
    x_test = {}
    for key in causal_net_Data.x_preprocess:
        x_train[key] = causal_net_Data.x_preprocess[key][training_idx]
        x_test[key] = causal_net_Data.x_preprocess[key][test_idx]
            
    treatment_train, treatment_test = treatment_data[training_idx], treatment_data[test_idx]
    weight_train, weight_test = weight_data[training_idx], weight_data[test_idx]
    
    target_train, target_test = target_data[training_idx], target_data[test_idx]
        
    target_treat_train = np.concatenate([target_train, treatment_train],1)

    metrics = [tweedie_loss, binary_classification_loss, treatment_accuracy, track_epsilon]

    if targeted_regularization:
        loss = targeted_regularisation(ratio=ratio, dragonnet_loss=knob_loss)
    else:
        loss = knob_loss
        
    import time
    start_time = time.time()
    
    print("\n---------------------------------------------------------\n Starting to train DragonNet stage 1 !!!!!!\n")
    
    if embeddings_path:
        print('Using embeddings specified in `embeddings_path`')
        inputs, output = emdedding_net_skeleton(causal_net_Data, use_pretrained_model_weights=True, embeddings_path=embeddings_path)
    else:
        print(' Embeddings will be trained from scratch!!!!!.')
        inputs, output = emdedding_net_skeleton(causal_net_Data)
    model = dragonnet_with_embeddings(inputs, output, 0.01)
    
    model.compile(
        optimizer=Adam(learning_rate=adam_lr, 
                 beta_1=adam_b1, 
                 beta_2=adam_b2, 
                 epsilon=adam_eps,
                 amsgrad=adam_amsgrad),
        loss=loss, 
        metrics=metrics)
    
    adam_checkpoint = checkpoint_parent/'adam_checkpoint'
    
    adam_callbacks = [
        TrainPerformanceViz(),
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', mode='min', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=verbose, mode='auto',
                          min_delta=1e-8, cooldown=0, min_lr=0),
        ModelCheckpoint(
            filepath=adam_checkpoint/'model.{epoch:02d}-{val_loss:.2f}', monitor='val_loss', mode='min', verbose=0, save_best_only=False,
            save_weights_only=False, save_freq='epoch')]
    
    model.fit(x_train, 
                  target_treat_train,
                  callbacks=adam_callbacks,
                  validation_split=val_split,
                  epochs=rd_one_epoch,
                  batch_size=rd_one_batch, 
                  verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** Stage 1 has run for  ", elapsed_time, " seconds")
    
    print("\n---------------------------------------------------------\n Starting to train DragonNet stage 2 !!!!!!!!\n")       

    sig_optimizer = SGD(learning_rate=sgd_lr, momentum=sgd_momentum, nesterov=sgd_nesterov)

    model.compile(optimizer=sig_optimizer, 
                      loss=loss, 
                      metrics=metrics)
    
    sgd_checkpoint = checkpoint_parent/'sgd_checkpoints/'
    sgd_callbacks = [
        TrainPerformanceViz(),
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=2, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=verbose, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0),
        ModelCheckpoint(
            filepath=sgd_checkpoint/'model.{epoch:02d}-{val_loss:.2f}', monitor='val_loss', mode='min', verbose=0, save_best_only=False,
            save_weights_only=False, save_freq='epoch')
                    ]

    model.fit(x_train, 
                  target_treat_train, 
                  callbacks=sgd_callbacks,
                  validation_split=val_split,
                  epochs=rd_two_epoch,
                  batch_size=rd_two_batch, 
                  verbose=verbose)

    elapsed_time = time.time() - start_time
    print("***************************** DragonNet training has run for ", elapsed_time, " seconds")

    target_treat_prediction_test = model.predict(x_test)
    target_treat_prediction_train = model.predict(x_train)

    train_results = []
    test_results = []
    
    test_results += [output_process(target_treat_prediction_test, treatment_test, target_test, target_scaler, x_test, test_idx, weight_test)]
    train_results += [output_process(target_treat_prediction_train, treatment_train, target_train, target_scaler, x_train, training_idx, weight_train)]
    K.clear_session()

    return test_results, train_results, model

def unpack(output_dict, causal_net_data, user_cat_level_definition=None):
    '''
    Helper function for unpacking dragonnet model results. 
    Pass a levels_dict: Emblem levels will be named.
    Don't pass levels_dict: original Emblem data levels returned.)
    '''
    categorical_variables = causal_net_data.categorical_variables
    continuous_variables = causal_net_data.continuous_variables
    categorical_level_dict = causal_net_data.categorical_level_dict
    
    ## Fetching results 
    results = {k:v for k, v in output_dict[0].items() if k in ['q_t0','q_t1','g','t','weight']}
    results['t'] = results['t'].reshape(-1,1)
    results['g'] = results['g'].reshape(-1,1)
    for key in results:
        results[key] = results[key][:,0]
    results_df = pd.DataFrame.from_dict(results)
    
    ## Extract categoricals
    categorical_dict = {k:v for k,v in output_dict[0]['x'].items() if k!='cont_vars'}
    categorical_dict['y'] = output_dict[0]['y']
    for key in categorical_dict:
        categorical_dict[key] = np.transpose(categorical_dict[key])[0]
    categorical_df = pd.DataFrame.from_dict(categorical_dict).replace(categorical_level_dict)
    
    ## Extract continuous variables
    continuous_df = pd.DataFrame(output_dict[0]['x']['cont_vars'])
    continuous_df.columns = continuous_variables
    
    final_df = pd.concat([results_df, categorical_df, continuous_df], axis=1)
    
    if user_cat_level_definition:
        final_df.loc[:,categorical_variables+continuous_variables] = final_df.loc[:,categorical_variables+continuous_variables].astype('str')
        final_df = final_df.replace(user_cat_level_definition)
    
    return final_df
