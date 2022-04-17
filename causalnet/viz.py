import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def fetch_model_var_info(model, dragon_data):
    '''
    Function to extract variable names and predicted embeddings of the categorical variables from an embeddings model.
    '''
    cat_vars = dragon_data.categorical_variables
    cont_vars = dragon_data.continuous_variables
    return dict(zip(cat_vars+cont_vars, model.get_weights()[:len(cat_vars)+len(cont_vars)]))

def embeddings_to_pandas(variable, embedding_dictionary, embed_json):
    '''
    Function to convert embeddings of a variable into a pandas df.
    '''
    np_frame = np.transpose(embedding_dictionary[variable])
    number_embeddings = np_frame.shape[0]    
    number_levels = [embed_json[variable][i] for i in embed_json[variable]]
    append_dict = {}
    append_dict[variable] = number_levels
    for i in range(number_embeddings):
        append_dict['embedding_'+str(i)] = np_frame[i]
    append_frame = pd.DataFrame(append_dict)
    # append_frame[variable] = append_frame[variable].astype('str').map(dict_of_dicts[variable])
    append_frame[variable] = append_frame[variable].astype('str')

    return append_frame
        
def fetch_cat_levels(variable, factored_dict):
    '''
    Function to get categorical levels
    '''
    value_list = list(factored_dict[variable].values())
    # str_list = [str(i) for i in value_list]
    
    # return [level_dict[variable].get(item, item) for item in str_list]
    return value_list
        
def embeddings_viz(variable, model, dragon_data, figsize=(6,6), pca_size=3):
    '''
    Function to plot embeddings
    '''
    embedding_dictionary = fetch_model_var_info(model, dragon_data)
    factored_dict = dragon_data.categorical_level_dict
    plot_dat = embeddings_to_pandas(variable, embedding_dictionary, factored_dict)

    fig = plt.figure()
    fig = plt.figure(figsize=figsize)
    fig = plt.title(variable)
    
    if plot_dat.shape[1] > 4:
        level_list = fetch_cat_levels(variable, factored_dict)
        
        if pca_size==3:
            pca = PCA(n_components=3)
            data = pca.fit_transform(embedding_dictionary[variable])
            x, y, z = data[:,0], data[:,1], data[:,2]
            ax = plt.axes(projection='3d')
            ax.set_xlabel('pca 0')
            ax.set_ylabel('pca 1')
            ax.set_zlabel('pca 2')
            for i in range(len(level_list)):
                xi, yi, zi = x[i], y[i], z[i]
                ax.scatter(xi,yi,zi)
                ax.text(xi,yi,zi,'{0}'.format(level_list[i]))
        elif pca_size==2:
            pca = PCA(n_components=2)
            data = pca.fit_transform(embedding_dictionary[variable])
            x, y = data[:,0], data[:,1]
            fig = plt.xlabel('pca 0')
            fig = plt.ylabel('pca 1')
            for i in range(len(level_list)):
                xi, yi = x[i], y[i]
                plt.scatter(xi,yi)
                plt.text(xi,yi,'{0}'.format(level_list[i]))
    elif plot_dat.shape[1] == 4:
        ax = plt.axes(projection='3d')
        ax.set_xlabel('embedding 0')
        ax.set_ylabel('embedding 1')
        ax.set_zlabel('embedding 2')
        for i in range(plot_dat.shape[0]):
            x, y, z = plot_dat.iloc[i][1], plot_dat.iloc[i][2], plot_dat.iloc[i][3]
            ax.scatter(x,y,z)
            ax.text(x,y,z,'{0}'.format(plot_dat[variable][i]))
    elif plot_dat.shape[1] < 4:
        fig = plt.xlabel('embedding 0')
        fig = plt.ylabel('embedding 1')
        for i, label in enumerate(plot_dat[variable]):
            x, y = plot_dat.iloc[i][1], plot_dat.iloc[i][2]
            plt.scatter(x, y)
            plt.annotate(label, (x,y))
    return None

def plot_lift_curve(prediction, actual, weight, n_bins = 10, plot = True, norm = True, disagree = 'abs', **kwargs):
    
    """
    Function that returns a single-lift dataset with option to plot if argument `plot` set to True

    Arguments:
        n_bins (integer): Number of model disagreement bins for lift chart

        plot (boolean): Whether to return a plot of the double-lift dataset

        norm (boolean): Set to True if you want to normalize the lift-curve dataset such that each
        bin is divided by the weighted average actual loss

        disagree (string): Either 'pct' or 'abs'. If 'pct' model disagreement is based
        on percentage difference of the two model predictions, (pred / act). If 'abs' then 
        model disagreement is based on the absolute difference between the two model 
        predictions, (pred - act)

        **kwargs: Plotting argument to pass to matplotlib. If specific colors are desired
        then provide as list of length 3. The first color will be used to plot the
        prediction. The second color will plot the the actual, and the third color will 
        plot the percent of records in each model disagreement bin.
    """
    figsize = kwargs.get('figsize', (7,5))
    style = kwargs.get('style', '-o')
    title = kwargs.get('title', 'Single Lift-Chart')
    colors = kwargs.get('color', ['blue', 'orange', 'black'])
    grid = kwargs.get('gird', False)
    ms = kwargs.get('ms', 7)
    linewidth = kwargs.get('linewidth', 2)
    d = pd.DataFrame({'p': prediction, 'a': actual,
                        'w': weight})

    d['pw'] = d.p * d.w
    d['aw'] = d.a * d.w

    x_label = 'Model Disagreement Bin for (Pred / Act)'
    if disagree == 'pct':
        d['disagree'] = d.p / d.a
    elif disagree == 'abs':
        d['disagree'] = d.p - d.a
        x_label = 'Model Disagreement Bin for (New - Old)'
    else:
        print(("One of 'pct' and 'abs' are expected for `disagree`. Going to use the "
                "percentage disagreement for constructing the lift chart"))
        d['disagree'] = d.p / d.a

    d['bin'] = pd.qcut(d.disagree, q = n_bins, labels = False, duplicates = 'drop')

    g = d.groupby('bin')
    g_bin = g['pw', 'aw', 'w'].sum().reset_index()
    g_bin['records'] = g.size()
    
    if norm:
        g_bin['pred'] = g_bin.pw / g_bin.aw
        g_bin['act'] = 1.0
    else:
        g_bin['pred'] = g_bin.pw / g_bin.w
        g_bin['act'] = g_bin.aw / g_bin.w
    
    g_bin['bin_pct'] = g_bin.records/g_bin.records.sum()

    if plot:
        ax1 = g_bin.plot(x = 'bin', y = ['pred','act'], style = '-o',  
                            ms = ms, color = colors[:2], title = title, figsize = figsize,
                            linewidth = linewidth)
        ax2 = g_bin.plot(kind = 'bar', ax = ax1, y = 'bin_pct', x = 'bin', 
                            color = colors[2], sharex = True, secondary_y = True,
                            alpha = 0.3)
        ax2.set_ylim(0.0, (1/n_bins)*2.4)
        ax1.set_xlabel(xlabel = x_label)
        fig = ax1.get_figure()
        return fig
        
def plot_double_lift_curve(valuer, n_bins = 10, plot = True, norm = True, disagree = 'pct', **kwargs):
    """
    Function that returns a double-lift dataset with option to plot if argument `plot` set to True

    Arguments:
        n_bins (integer): Number of model disagreement bins for the double-lift chart

        plot (boolean): Whether to return a plot of the double-lift dataset

        norm (boolean): Whether to normalize the double-lift dataset such that  each
        bin is divided by the weighted average actual loss

        disagree (string): Either 'pct' or 'abs'. If 'pct' model disagreement is based
        on percentage difference of the two model predictions, (new / old). If 'abs' then 
        model disagreement is based on the absolute difference between the two model 
        predictions, (new - old)

        **kwargs: Plotting argument to pass to matplotlib. If specific colors are desired
        then provide as list of length 4. The first color will be used to plot the old
        prediction. The second color will plot the new prediction. The third color will plot
        the actual, and the fourth color will plot the percent of records in each model
        disagreement bin
    """
    figsize = kwargs.get('figsize', (7,5))
    style = kwargs.get('style', '-o')
    title = kwargs.get('title', 'Double Lift-Chart')
    colors = kwargs.get('color', ['blue', 'orange', 'black', 'gold'])
    grid = kwargs.get('gird', False)
    ms = kwargs.get('ms', 7)
    linewidth = kwargs.get('linewidth', 2)
    d = pd.DataFrame({'p0': valuer.pred0, 'p1': valuer.pred1, 'a': valuer.loss,
                        'w': valuer.pol0})

    d['pw0'] = d.p0 * d.w
    d['pw1'] = d.p1 * d.w
    d['aw'] = d.a * d.w

    x_label = 'Model Disagreement Bin for (New / Old)'
    if disagree == 'pct':
        d['disagree'] = d.p1 / d.p0
    elif disagree == 'abs':
        d['disagree'] = d.p1 - d.p0
        x_label = 'Model Disagreement Bin for (New - Old)'
    else:
        print(("One of 'pct' and 'abs' are expected for `disagree`. Going to use the "
                "percentage disagreement for consturcting the lift chart"))
        d['disagree'] = d.p1 / d.p0
    # to use weights would take too long and not be much value add
    d['bin'] = pd.qcut(d.disagree, q = n_bins, labels = False, duplicates = 'drop')

    g = d.groupby('bin')
    g_bin = g['pw0', 'pw1', 'aw', 'w'].sum().reset_index()
    g_bin['records'] = g.size()
    
    if norm:
        g_bin['old_pred'] = g_bin.pw0 / g_bin.aw
        g_bin['new_pred'] = g_bin.pw1 / g_bin.aw
        g_bin['act'] = 1.0
    else:
        g_bin['old_pred'] = g_bin.pw0 / g_bin.w
        g_bin['new_pred'] = g_bin.pw1 / g_bin.w
        g_bin['act'] = g_bin.aw / g_bin.w
    
    g_bin['bin_pct'] = g_bin.records/g_bin.records.sum()

    if plot:
        ax1 = g_bin.plot(x = 'bin', y = ['old_pred', 'new_pred'], style = '-o',  
                            ms = ms, color = colors[:2], title = title, figsize = figsize,
                            linewidth = linewidth)
        ax2 = g_bin.plot(ax = ax1, x = 'bin', y = 'act', color = colors[2], linestyle = '--',
                            linewidth = linewidth)
        ax3 = g_bin.plot(kind = 'bar', ax = ax1, y = 'bin_pct', x = 'bin', 
                            color = colors[3], sharex = True, secondary_y = True,
                            alpha = 0.3)
        ax3.set_ylim(0.0, (1/n_bins)*2.4)
        ax1.set_xlabel(xlabel = x_label)
        fig = ax1.get_figure()
        return fig