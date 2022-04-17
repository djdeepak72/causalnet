import pandas as pd
import numpy as np
from scipy.special import logit, expit
from scipy.optimize import minimize
from .utilities import truncate_by_g, mse, cross_entropy, truncate_all_by_g

# robust ATT estimator described in eqn 3.9 of
# https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf
def ate_aiptw(q_t0, q_t1, g, t, y, prob_t):
    att_aiptw = psi_aiptw(q_t0, q_t1, g, t, y, prob_t)
    atnott_aiptw = psi_aiptw(q_t1, q_t0, 1.-g, 1-t, y, 1.-prob_t)
    ate_aiptw = att_aiptw*prob_t + atnott_aiptw*(1.-prob_t)
    return ate_aiptw

def ate_psi(q_t0, q_t1, g, t, y, prob_t):
    att_psi = psi_plugin(q_t0, q_t1, g, t, y, prob_t)
    atnott_psi = psi_plugin(q_t1, q_t0, 1.-g, 1-t, y, 1.-prob_t)
    ate_psi = att_psi*prob_t + atnott_psi*(1.-prob_t)
    return(ate_psi)

def att_estimates(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):

    one_step_tmle = make_one_step_tmle(prob_t, deps_default=0.0001)

    very_naive = psi_very_naive(q_t0, q_t1, g, t, y, truncate_level)
    q_only = psi_q_only(q_t0, q_t1, g, t, y, truncate_level)
    plugin = psi_plugin(q_t0, q_t1, g, t, y, prob_t, truncate_level)
    aiptw = psi_aiptw(q_t0, q_t1, g, t, y, prob_t, truncate_level)
    one_step_tmle = one_step_tmle(q_t0, q_t1, g, t, y, truncate_level)  # note different signature

    estimates = {'very_naive': very_naive, 'q_only': q_only, 'plugin': plugin, 'one_step_tmle': one_step_tmle, 'aiptw': aiptw}

    return estimates

def treatment_effects(df, groupby_cols, inc_raw=True, cut_size=1000, prop='g', wgt='weight', resp='y', treat='t'):
    '''
    * This function outputs a dataFrame that has selection effect(ie. percentage value that the treated group is better than untreated) by the subgroup specified by the `groupby_cols`.
    * The input df should be the output of the `unpack` function
    * groupby_cols can be a single column passed as a single column name or as a list of column names.
    * If inc_raw is True (default), the target variable predictions given the treatment group (ie. y_t1 and y_t0) will be returned. Otherwise, only selection effect as a percentage will be returned.
    * Control the number of propensity score groups using the cut_size parameter. 
    * prop, wgt, resp, and treat arguments are to be used if the column names for propensity, weight, response, and treatment differ from the default names used in this package.
    * Treatment column should be only of 0 or 1 values.
    * The `selection` column is as a percentage value that the treated group performed better than the untreated group (For example, selection of -0.1 means treated group perform 10% better than the untreated group)
    '''
    
    ## Get initial df
    output_df = df.copy()
    wgtd_mean = lambda x: np.average(x, weights=output_df.loc[x.index, wgt])
    
    output_df['prop_cut'] = pd.cut(df[prop], cut_size, labels=False)
    output_df['wgtd_y'] = df[resp] * df[wgt]
    
    treated = output_df[output_df[treat] == 1].rename(columns={'y':'y_t1','weight':'weight_t1'})
    untreated = output_df[output_df[treat] == 0].rename(columns={'y':'y_t0','weight':'weight_t0'})
    
    t1_group = treated.groupby('prop_cut').agg({'y_t1':wgtd_mean,
                           'weight_t1':'sum'}).reset_index()
    t0_group = untreated.groupby('prop_cut').agg({'y_t0':wgtd_mean,
                               'weight_t0':'sum',
                               'wgtd_y':'sum'}).reset_index()
    test_df = pd.merge(treated, t0_group[['y_t0','weight_t0','prop_cut']], on='prop_cut',how='left')
    
    ## Get selection df
    if type(groupby_cols)=='str':
        groupby_cols = [groupby_cols]
    wgtd_mean = lambda x: np.average(x, weights=test_df.loc[x.index, 'weight_t1'])
    out_df = test_df.groupby(groupby_cols).agg({'y_t1':wgtd_mean,'y_t0':'mean','weight_t1':'sum','weight_t0':'sum'}).reset_index()
    out_df['selection'] = (out_df.y_t1 - out_df.y_t0)/out_df.y_t0
    if not inc_raw:
        out_df = out_df.drop(columns=['y_t1','y_t0','weight_t1','weight_t0'])
    return out_df

def _perturbed_model(q_t0, q_t1, g, t, q, eps):
    # helper function for psi_tmle

    h1 = t / q - ((1 - t) * g) / (q * (1 - g))
    full_q = (1.0 - t) * q_t0 + t * q_t1
    perturbed_q = full_q - eps * h1

    def q1(t_cf, epsilon):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q - epsilon * h_cf

    psi_init = np.mean(t * (q1(np.ones_like(t), eps) - q1(np.zeros_like(t), eps))) / q
    h2 = (q_t1 - q_t0 - psi_init) / q
    perturbed_g = expit(logit(g) - eps * h2)

    return perturbed_q, perturbed_g


def psi_tmle(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    """
    Near canonical van der Laan TMLE, except we use a
    1 dimension epsilon shared between the Q and g update models
    """

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    def _perturbed_loss(eps):
        pert_q, pert_g = _perturbed_model(q_t0, q_t1, g, t, prob_t, eps)
        loss = (np.square(y - pert_q)).mean() + cross_entropy(t, pert_g)
        return loss

    eps_hat = minimize(_perturbed_loss, 0.)
    eps_hat = eps_hat.x[0]

    def q2(t_cf, epsilon):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q - epsilon * h_cf

    psi_tmle = np.mean(t * (q2(np.ones_like(t), eps_hat) - q2(np.zeros_like(t), eps_hat))) / prob_t
    return psi_tmle


def make_one_step_tmle(prob_t, deps_default=0.001):
    "Make a function that computes the 1-step TMLE ala https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4912007/"

    def _perturb_q(q_t0, q_t1, g, t, deps=deps_default):
        h1 = t / prob_t - ((1 - t) * g) / (prob_t * (1 - g))

        full_q = (1.0 - t) * q_t0 + t * q_t1
        perturbed_q = full_q - deps * h1
        # perturbed_q= expit(logit(full_q) - deps*h1)
        return perturbed_q

    def _perturb_g(q_t0, q_t1, g, deps=deps_default):
        h2 = (q_t1 - q_t0 - _psi(q_t0, q_t1, g)) / prob_t
        perturbed_g = expit(logit(g) - deps * h2)
        return perturbed_g

    def _perturb_g_and_q(q0_old, q1_old, g_old, t, deps=deps_default):
        # get the values of Q_{eps+deps} and g_{eps+deps} by using the recursive formula

        perturbed_g = _perturb_g(q0_old, q1_old, g_old, deps=deps)

        perturbed_q = _perturb_q(q0_old, q1_old, perturbed_g, t, deps=deps)
        perturbed_q0 = _perturb_q(q0_old, q1_old, perturbed_g, np.zeros_like(t), deps=deps)
        perturbed_q1 = _perturb_q(q0_old, q1_old, perturbed_g, np.ones_like(t), deps=deps)

        return perturbed_q0, perturbed_q1, perturbed_q, perturbed_g

    def _loss(q, g, y, t):
        # compute the new loss
        q_loss = mse(y, q)
        g_loss = cross_entropy(t, g)
        return q_loss + g_loss

    def _psi(q0, q1, g):
        return np.mean(g*(q1 - q0)) / prob_t

    def tmle(q_t0, q_t1, g, t, y, truncate_level=0.05, deps=deps_default):
        """
        Computes the tmle for the ATT (equivalently: direct effect)
        :param q_t0:
        :param q_t1:
        :param g:
        :param t:
        :param y:
        :param truncate_level:
        :param deps:
        :return:
        """
        q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

        eps = 0.0

        q0_old = q_t0
        q1_old = q_t1
        g_old = g

        # determine whether epsilon should go up or down
        # translated blindly from line 299 of https://github.com/cran/tmle/blob/master/R/tmle.R
        h1 = t / prob_t - ((1 - t) * g) / (prob_t * (1 - g))
        full_q = (1.0 - t) * q_t0 + t * q_t1
        deriv = np.mean(prob_t*h1*(y-full_q) + t*(q_t1 - q_t0 - _psi(q_t0, q_t1, g)))
        if deriv > 0:
            deps = -deps

        # run until loss starts going up
        # old_loss = np.inf  # this is the thing used by Rose' implementation
        old_loss = _loss(full_q, g, y, t)

        while True:
            # print("Psi: {}".format(_psi(q0_old, q1_old, g_old)))

            perturbed_q0, perturbed_q1, perturbed_q, perturbed_g = _perturb_g_and_q(q0_old, q1_old, g_old, t, deps=deps)

            new_loss = _loss(perturbed_q, perturbed_g, y, t)

            # check if converged
            if new_loss > old_loss:
                if eps == 0.:
                    print("Warning: no update occurred (is deps too big?)")
                return _psi(q0_old, q1_old, g_old), eps
            else:
                eps += deps

                q0_old = perturbed_q0
                q1_old = perturbed_q1
                g_old = perturbed_g

                old_loss = new_loss

    return tmle


def psi_q_only(q_t0, q_t1, g, t, y, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    ite_t = (q_t1 - q_t0)[t == 1]
    estimate = ite_t.mean()
    return estimate


def psi_plugin(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    ite_t = g*(q_t1 - q_t0)/prob_t
    estimate = ite_t.mean()
    return estimate


def psi_aiptw(q_t0, q_t1, g, t, y, prob_t, truncate_level=0.05):
    # the robust ATT estimator described in eqn 3.9 of
    # https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)
    estimate = (t*(y-q_t0) - (1-t)*(g/(1-g))*(y-q_t0)).mean() / prob_t

    return estimate

def _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps):
    """
    Helper function for psi_tmle_bin_outcome
    Returns q_\eps (t,x)
    (i.e., value of perturbed predictor at t, eps, x; where q_t0, q_t1, g are all evaluated at x
    """
    h = t * (1./g) - (1.-t) / (1. - g)
    full_lq = (1.-t)*logit(q_t0) + t*logit(q_t1)  # logit predictions from unperturbed model
    logit_perturb = full_lq + eps * h
    return expit(logit_perturb)


def psi_tmle_bin_outcome(q_t0, q_t1, g, t, y, truncate_level=0.05):
    # solve the perturbation problem

    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)

    eps_hat = minimize(lambda eps: cross_entropy(y, _perturbed_model_bin_outcome(q_t0, q_t1, g, t, eps))
                       , 0., method='Nelder-Mead')

    eps_hat = eps_hat.x[0]

    def q1(t_cf):
        return _perturbed_model_bin_outcome(q_t0, q_t1, g, t_cf, eps_hat)

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    return np.mean(ite)


def psi_tmle_cont_outcome(q_t0, q_t1, g, t, y, eps_hat=None, truncate_level=0.05):
    q_t0, q_t1, g, t, y = truncate_all_by_g(q_t0, q_t1, g, t, y, truncate_level)


    g_loss = mse(g, t)
    h = t * (1.0/g) - (1.0-t) / (1.0 - g)
    full_q = (1.0-t)*q_t0 + t*q_t1 # predictions from unperturbed model

    if eps_hat is None:
        eps_hat = np.sum(h*(y-full_q)) / np.sum(np.square(h))

    def q1(t_cf):
        h_cf = t_cf * (1.0 / g) - (1.0 - t_cf) / (1.0 - g)
        full_q = (1.0 - t_cf) * q_t0 + t_cf * q_t1  # predictions from unperturbed model
        return full_q + eps_hat * h_cf

    ite = q1(np.ones_like(t)) - q1(np.zeros_like(t))
    psi_tmle = np.mean(ite)

    # standard deviation computation relies on asymptotic expansion of non-parametric estimator, see van der Laan and Rose p 96
    ic = h*(y-q1(t)) + ite - psi_tmle
    psi_tmle_std = np.std(ic) / np.sqrt(t.shape[0])
    initial_loss = np.mean(np.square(full_q-y))
    final_loss = np.mean(np.square(q1(t)-y))

    print("tmle epsilon_hat: ", eps_hat)
    print("initial risk: {}".format(initial_loss))
    print("final risk: {}".format(final_loss))

    return psi_tmle, psi_tmle_std, eps_hat, initial_loss, final_loss, g_loss


def psi_iptw(q_t0, q_t1, g, t, y, truncate_level=0.05):
    ite=(t / g - (1-t) / (1-g))*y
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    ite = (q_t1 - q_t0)
    return np.mean(truncate_by_g(ite, g, level=truncate_level))


def psi_very_naive(q_t0, q_t1, g, t, y, truncate_level=0.):
    return y[t == 1].mean() - y[t == 0].mean()


def ates_from_atts(q_t0, q_t1, g, t, y, truncate_level=0.05):
    """
    Sanity check code: ATE = ATT_1*P(T=1) + ATT_0*P(T=1)
    :param q_t0:
    :param q_t1:
    :param g:
    :param t:
    :param y:
    :param truncate_level:
    :return:
    """

    prob_t = t.mean()

    att = att_estimates(q_t0, q_t1, g, t, y, prob_t, truncate_level=truncate_level)
    atnott = att_estimates(q_t1, q_t0, 1.-g, 1-t, y, 1.-prob_t, truncate_level=truncate_level)

    att['one_step_tmle'] = att['one_step_tmle'][0]
    atnott['one_step_tmle'] = atnott['one_step_tmle'][0]

    ates = {}
    for k in att.keys():
        ates[k] = att[k]*prob_t + atnott[k]*(1.-prob_t)

    return ates


def main():
    pass


if __name__ == "__main__":
    main()