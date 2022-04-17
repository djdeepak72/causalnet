import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

from .metrics import rmse, tweedie_loss



def categorical_level_mapper(data, cat_var_list):
    '''
    Maps all unique levels of categorical variable into a dictionary
    '''
    categorical_level_map = {}
    for var in cat_var_list:
        unique_level_dict = {}
        unique_list = data[var].unique()
        unique_level_dict = dict(enumerate(unique_list))
        categorical_level_map[var] = {int(k):v for k,v in unique_level_dict.items()}
    return(categorical_level_map)

class CausalNetData:
    ''' Class object that holds data after preprocessing'''
    def __init__(self):
        self.x_preprocess = {}
        self.treatment_preprocess = None
        self.target_preprocess = None
        self.target_scaler = None
        self.categorical_variables = None
        self.categorical_size_dict = None
        self.categorical_embedding_size = None
        self.continuous_variables = None
        self.weight_preprocess = None
        self.training_index = None
        self.test_index = None
        self.categorical_level_dict = {}

def fetch_embedding_weights(embedding_location, categorical_list, total_variables):
    '''
    Gets the embedding weights from saved embedding model
    '''
    embedding_weight_list = load_model(embedding_location, custom_objects = {'rmse': rmse, 'tweedie_loss':tweedie_loss}).get_weights()[:total_variables]
    embedding_weights = {}
    i = 0
    for c in categorical_list:
        embedding_weights[c] = embedding_weight_list[i]
        i+=1
    return(embedding_weights)

def fetch_cat_embedding_info(dataframe, categorical_list):
    """
    Structure:
        dataframe = dataframe you'll be using for the analysis
        categorical_list = list of all the categorical variables in the dataframe 
    Returns:
        Returns categorical_size_dict and categorical_embedding_size, which are both dictionaries that hold, respectively, the number of unique levels and embedding size for each categorical variable.
    """
    categorical_size_dict = {}
    categorical_embedding_size = {}

    for c in categorical_list:
        categorical_size_dict[c] = dataframe[c].nunique()
        categorical_embedding_size[c] = min(50, categorical_size_dict[c]//2+1)
    
    return (categorical_size_dict, categorical_embedding_size)


def fetch_global_cat_embedding_info(categorical_list, user_cat_level_definition):
    """
    Creates embedding sizes that are dependent on the predefined, expected number of a levels for the categorical variables. 
    /nThis is necessary for data that may be missing data in some partitions. level_defn variable must be defined as a dictionary where the variables in the model
    /nare the keys, and the nested dictionaries are the length of the desired variable definition. Level to name dictionaries work particularly well.
    """
    
    categorical_size_dict = {}
    categorical_embedding_size = {}
    
    for c in categorical_list:
        categorical_size_dict[c] = len(user_cat_level_definition[c])
        categorical_embedding_size[c] = min(50, categorical_size_dict[c]//2+1)
        
    return(categorical_size_dict, categorical_embedding_size)


def preprocessing(data, categorical_variables, continuous_variables, target_variable, weight_variable=None, treatment_variable=None, scale_y = False, weight_target = False, train_split=.8, random_state=11, user_cat_level_definition = None, y_clip = None):
    """
    All _var passes should be as lists. 
    /n The quant_clip will allow you to clip extreme values of your data that may impact your normalization process. Default is None.
    """
    causal_inference_df = CausalNetData()
    
    ############################
    # Define cat and cont cols #
    ############################
    x_preprocess_dict = {}
    for var in categorical_variables:
        x_data = pd.DataFrame()
        if user_cat_level_definition:
            x_data[var] = data[var] - 1
        else:
            x_data[var] = data[var].factorize()[0]
        x_preprocess_dict[var] = np.asarray(x_data)
    x_preprocess_dict['cont_vars'] = np.asarray(data[continuous_variables])
    
    #######################
    # Define conditionals #
    #######################
    weight_column = data[weight_variable].values
    if weight_target:
        weight_target_col = np.asarray(data[weight_variable])
        weighted_target = data[target_variable].values * weight_target_col
    else:
        weighted_target = data[target_variable].values
    
    if treatment_variable:
        treatment_column = data[treatment_variable].values
    else:
        treatment_column = None
    
    if y_clip:
        print("Setting max value of target variable at {:.2f}".format(y_clip))
        weighted_target = np.clip(weighted_target,None,y_clip)
    
    if scale_y:
        print("\nNormalizing your y, please hold.\n")
        target_scaler = MinMaxScaler().fit(weighted_target)
        target_preprocess = target_scaler.transform(weighted_target)
    else:
        target_scaler = None
        target_preprocess = weighted_target
    
    if user_cat_level_definition:
        categorical_size_dict, categorical_embedding_size = fetch_global_cat_embedding_info(categorical_variables, user_cat_level_definition)
        categorical_level_dict = {}
        for i in user_cat_level_definition:
            inner_dict = {}
            for j in user_cat_level_definition[i]:
                inner_dict[j] = j
            categorical_level_dict[i] = inner_dict
    else:
        categorical_level_dict = categorical_level_mapper(data, categorical_variables)
        categorical_size_dict, categorical_embedding_size = fetch_cat_embedding_info(data, categorical_variables)    
    ###############################
    # Define train and test split #
    ###############################     
    training_idxs, testing_idxs = train_test_split(np.arange(data.values.shape[0]), train_size=train_split, random_state=random_state)
    
    ###################################
    # Define returnable dragon object #
    ###################################
    causal_inference_df.x_preprocess = x_preprocess_dict
    causal_inference_df.treatment_preprocess = treatment_column
    causal_inference_df.target_preprocess = target_preprocess
    causal_inference_df.target_scaler  = target_scaler
    causal_inference_df.categorical_variables = categorical_variables
    causal_inference_df.categorical_size_dict = categorical_size_dict
    causal_inference_df.categorical_embedding_size = categorical_embedding_size
    causal_inference_df.continuous_variables = continuous_variables
    causal_inference_df.weight_preprocess = weight_column
    causal_inference_df.training_index = training_idxs
    causal_inference_df.test_index = testing_idxs
    causal_inference_df.categorical_level_dict = categorical_level_dict
    
    return causal_inference_df
