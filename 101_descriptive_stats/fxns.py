import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
import warnings
from scipy.stats import binned_statistic
from numpy import log10, sqrt
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from scipy.stats.kde import gaussian_kde

#########################################################################################
################################ FUNCTIONS ##############################################

def get_kdens_choose_kernel(_list, kernel):
    """ Finds the kernel density function across a sample of SADs """
    density = gaussian_kde(_list)
    n = len(_list)
    #xs = np.linspace(0, 1, n)
    xs = np.linspace(min(_list), max(_list), n)
    density.covariance_factor = lambda : kernel
    density._compute_covariance()
    D = [xs,density(xs)]
    return D



def hulls(x,y):
    grain_p = 10
    clim_p = 95
    xran = np.arange(min(x), max(x), grain_p).tolist()
    binned = np.digitize(x, xran).tolist()
    bins = [list([]) for _ in range(len(xran))]

    for ii, val in enumerate(binned):
        bins[val-1].append(y[ii])

    pct5 = []
    pct95 = []
    xran2 = []

    for iii, _bin in enumerate(bins):
        if len(_bin) > 0:
            clim = clim_p
            pct5.append(np.percentile(_bin, 100 - clim))
            pct95.append(np.percentile(_bin, clim))
            xran2.append(xran[iii])

    return xran2, pct5, pct95
    

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Use of Euclidean distance in calculating geographical
    distance is rarely, if ever, recommended.
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371.0087714150598 * c
    return km



def randcolor():
    """
    Get a random rgb value (a random color).
    This is useful when plotting data for many classes,
    for example, when your plots have large legends and
    you don't want to have to prescribe a color for each class.
    """
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())
    
    
def get_mode(X, Y):
    
    xs = sorted(list(set(X)))
    ys = []
    
    for x in xs:
        yst = []
        for i, xi in enumerate(X):
            if xi == x:
                yi = Y[i]
                yst.append(yi)
        
        mode = stats.mode(yst, nan_policy='omit')
        mode = min(mode[0])
        ys.append(mode)
        
    return xs, ys


def get_hist(ls):
    
    xs = sorted(list(set(ls)))
    ys = []
    for x in xs:
        y = ls.count(x)
        ys.append(y)
        
    return [xs, ys]


def smoothing(x, y, f):
    lowess_frac = f  # size of data (%) for estimation =~ smoothing window
    lowess_it = 0
    x_smooth = x
    y_smooth = lowess(y, x, is_sorted=False, frac=lowess_frac, it=lowess_it, return_sorted=False)
    return x_smooth, y_smooth


def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    raw_data = np.array([x, y])
    x = np.array(x)
    y = np.array(y)
    raw_data = raw_data.transpose()
    
    # Get unique data points by adding each pair of points to a set
    unique_points = set()
    for xval, yval in raw_data:
        unique_points.add((xval, yval))
    
    count_data = []
    for a, b in unique_points:
        if logscale == 1:
            num_neighbors = len(x[((log10(x) - log10(a)) ** 2 +
                                   (log10(y) - log10(b)) ** 2) <= log10(radius) ** 2])
        else:        
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data



def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density
    
    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)
    
    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.
    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))
    
    if plot_obj == None:
        plot_obj = plt.axes()
        
    if loglog == 0:
        plot_obj.scatter(sorted_plot_data[:, 0], 
                         sorted_plot_data[:, 1],
                         c = np.log10(sorted_plot_data[:, 2]), 
                         s = 20, edgecolors='none')
    elif loglog == 'sqrt':
        plot_obj.scatter(np.sqrt(sorted_plot_data[:, 0]), 
                         np.sqrt(sorted_plot_data[:, 1]),
                         c = np.log10(sorted_plot_data[:, 2]), 
                         s = 20, edgecolors='none')
    return plot_obj



def obs_pred_rsquare(obs, pred):

    '''
    Determines the proportion of variability in a data set accounted for by a model
    In other words, this determines the proportion of variation explained by the 1:1 line
    in an observed-predicted plot.

    Used in various peer-reviewed publications:
        1. Locey, K.J. and White, E.P., 2013. How species richness and total abundance
        constrain the distribution of abundance. Ecology letters, 16(9), pp.1177-1185.
        2. Xiao, X., McGlinn, D.J. and White, E.P., 2015. A strong test of the maximum
        entropy theory of ecology. The American Naturalist, 185(3), pp.E70-E80.
        3. Baldridge, E., Harris, D.J., Xiao, X. and White, E.P., 2016. An extensive
        comparison of species-abundance distribution models. PeerJ, 4, p.e2823.
    '''

    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)
    

def difff(obs, exp):
    a_dif = np.abs(obs - exp)

    obs = np.abs(obs)
    exp = np.abs(exp)
    p_dif = float(100) * np.abs(obs-exp)/np.abs(np.mean([obs, exp]))
    p_err = float(100) * np.abs(obs-exp)/np.abs(exp)

    return p_err, p_dif, a_dif
    
    

def e_simpson(SAD): # based on 1/D, not 1 - D

    " Simpson's evenness "
    SAD = filter(lambda a: a != 0, SAD)

    D = 0.0
    N = sum(SAD)
    S = len(SAD)

    for x in SAD:
        D += (x*x) / (N*N)

    E = round((1.0/D)/S, 4)

    if E < 0.0 or E > 1.0:
        print 'Simpsons Evenness =',E
    return E


def skewness(RAD):
    skew = stats.skew(RAD)
    # log-modulo skewness
    lms = np.log10(np.abs(float(skew)) + 1)
    if skew < 0:
        lms = lms * -1
    return lms


def camargo(SAD): # function to calculate Camargo's eveness:
    S = len(SAD)
    SAD = np.array(SAD)/sum(SAD)
    SAD = SAD.tolist()

    E = 1
    for i in range(0, S-1):
        for j in range(i+1, S-1):

            pi = SAD[i]
            pj = SAD[j]
            E -= abs(pi - pj)/S

    return E


def EQ_evenness(SAD):

    SAD.reverse()
    S = len(SAD)

    y_list = np.log(SAD).tolist()
    x_list = []
    for rank in range(1, S+1):
        x_list.append((rank)/S)

    slope, intercept, rval, pval, std_err = stats.linregress(x_list, y_list)

    Eq = 1 + (-2/np.pi) * np.arctan(slope)
    return Eq


def simpsons_dom(SAD):
    D = 0.0
    N = sum(SAD)

    for x in SAD:
        D += x*(x-1)
    D = 1 - (D/(N*(N-1)))

    return D


def simpsons_evenness(SAD): # based on 1/D, not 1 - D
    D = 0.0
    N = sum(SAD)
    S = len(SAD)

    for x in SAD:
        D += (x*x) / (N*N)

    E = (1/D)/S

    return E


def e_var(SAD):
    P = np.log(SAD)
    S = len(SAD)
    X = 0
    for x in P:
        X += (x - np.mean(P))**2/S
    evar = 1 - 2/np.pi*np.arctan(X)
    return(evar)


def OE(SAD):
    S = len(SAD)
    N = sum(SAD)
    o = 0

    for ab in SAD:
        o += min(ab/N, 1/S)

    return o


def McNaughton(sad):
    sad.sort(reverse=True)
    return 100 * (sad[0] + sad[1])/sum(sad)



def Rlogskew(sad):
    S = len(sad)
    sad = np.log10(sad)
    mu = np.mean(sad)

    num = 0
    denom = 0
    for ni in sad:
        num += ((ni - mu)**3)/S
        denom += ((ni - mu)**2)/S

    t1 = num/(denom**(3/2))
    t2 = (S/(S - 2)) * np.sqrt((S - 1)/S)
    return t1 * t2






# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

#scikit-bio/skbio/diversity/alpha/_base.py
#Jai Ram Rideoutjairideout on Nov 19, 2014 subsample -> subsample_counts; subsample_items -> isubsample
# ----------------------------------------------------------------------------
# Copyright (c) 2013--, scikit-bio development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import sys
import os
from scipy.special import gammaln
from scipy.optimize import fmin_powell, minimize_scalar

mydir = os.path.expanduser("~/Desktop/Repos/rare-bio/tools/StatPak")
sys.path.append(mydir)
import SubSample
from SubSample import subsample_counts


def _validate(counts, suppress_cast=False):
    """Validate and convert input to an acceptable counts vector type.
    Note: may not always return a copy of `counts`!
    """
    counts = np.asarray(counts)

    if not suppress_cast:
        counts = counts.astype(int, casting='safe', copy=False)

    if counts.ndim != 1:
        raise ValueError("Only 1-D vectors are supported.")
    elif (counts < 0).any():
        raise ValueError("Counts vector cannot contain negative values.")

    return counts


def berger_parker_d(counts):
    """Calculate Berger-Parker dominance.
    Berger-Parker dominance is defined as the fraction of the sample that
    belongs to the most abundant OTUs:
    .. math::
       d = \\frac{N_{max}}{N}
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Berger-Parker dominance.
    Notes
    -----
    Berger-Parker dominance is defined in [1]_. The implementation here is
    based on the description given in the SDR-IV online manual [2]_.
    References
    ----------
    .. [1] Berger & Parker (1970). SDR-IV online help.
    .. [2] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    return counts.max() / counts.sum()


def brillouin_d(counts):
    """Calculate Brillouin index of alpha diversity, which is defined as:
    .. math::
       HB = \\frac{\\ln N!-\\sum^5_{i=1}{\\ln n_i!}}{N}
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Brillouin index.
    Notes
    -----
    The implementation here is based on the description given in the SDR-IV
    online manual [1]_.
    References
    ----------
    .. [1] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    nz = counts[counts.nonzero()]
    n = nz.sum()
    return (gammaln(n + 1) - gammaln(nz + 1).sum()) / n


def dominance(counts):
    """Calculate dominance.
    Dominance is defined as
    .. math::
       \\sum{p_i^2}
    where :math:`p_i` is the proportion of the entire community that OTU
    :math:`i` represents.
    Dominance can also be defined as 1 - Simpson's index. It ranges between
    0 and 1.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Dominance.
    See Also
    --------
    simpson
    Notes
    -----
    The implementation here is based on the description given in [1]_.
    References
    ----------
    .. [1] http://folk.uio.no/ohammer/past/diversity.html
    """
    counts = _validate(counts)
    freqs = counts / counts.sum()
    return (freqs * freqs).sum()


def doubles(counts):
    """Calculate number of double occurrences (doubletons).
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    int
        Doubleton count.
    """
    counts = _validate(counts)
    return (counts == 2).sum()


def enspie(counts):
    """Calculate ENS_pie alpha diversity measure.
    ENS_pie is equivalent to ``1 / dominance``.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        ENS_pie alpha diversity measure.
    See Also
    --------
    dominance
    Notes
    -----
    ENS_pie is defined in [1]_.
    References
    ----------
    .. [1] Chase and Knight (2013). "Scale-dependent effect sizes of ecological
       drivers on biodiversity: why standardised sampling is not enough".
       Ecology Letters, Volume 16, Issue Supplement s1, pgs 17-26.
    """
    counts = _validate(counts)
    return 1 / dominance(counts)


def equitability(counts, base=2):
    """Calculate equitability (Shannon index corrected for number of OTUs).
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    base : scalar, optional
        Logarithm base to use in the calculations.
    Returns
    -------
    double
        Measure of equitability.
    See Also
    --------
    shannon
    Notes
    -----
    The implementation here is based on the description given in the SDR-IV
    online manual [1]_.
    References
    ----------
    .. [1] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    numerator = shannon(counts, base)
    denominator = np.log(observed_otus(counts)) / np.log(base)
    return numerator / denominator


def esty_ci(counts):
    """Calculate Esty's CI.
    Esty's CI is defined as
    .. math::
       F_1/N \\pm z\\sqrt{W}
    where :math:`F_1` is the number of singleton OTUs, :math:`N` is the total
    number of individuals (sum of abundances for all OTUs), and :math:`z` is a
    constant that depends on the targeted confidence and based on the normal
    distribution.
    :math:`W` is defined as
    .. math::
       \\frac{F_1(N-F_1)+2NF_2}{N^3}
    where :math:`F_2` is the number of doubleton OTUs.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    tuple
        Esty's confidence interval as ``(lower_bound, upper_bound)``.
    Notes
    -----
    Esty's CI is defined in [1]_. :math:`z` is hardcoded for a 95% confidence
    interval.
    References
    ----------
    .. [1] Esty, W. W. (1983). "A normal limit law for a nonparametric
       estimator of the coverage of a random sample". Ann Statist 11: 905-912.
    """
    counts = _validate(counts)

    f1 = singles(counts)
    f2 = doubles(counts)
    n = counts.sum()
    z = 1.959963985
    W = (f1 * (n - f1) + 2 * n * f2) / (n ** 3)

    return f1 / n - z * np.sqrt(W), f1 / n + z * np.sqrt(W)


def fisher_alpha(counts):
    """Calculate Fisher's alpha.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Fisher's alpha.
    Raises
    ------
    RuntimeError
        If the optimizer fails to converge (error > 1.0).
    Notes
    -----
    The implementation here is based on the description given in the SDR-IV
    online manual [1]_. Uses ``scipy.optimize.minimize_scalar`` to find
    Fisher's alpha.
    References
    ----------
    .. [1] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    n = counts.sum()
    s = observed_otus(counts)

    def f(alpha):
        return (alpha * np.log(1 + (n / alpha)) - s) ** 2

    # Temporarily silence RuntimeWarnings (invalid and division by zero) during
    # optimization in case invalid input is provided to the objective function
    # (e.g. alpha=0).
    orig_settings = np.seterr(divide='ignore', invalid='ignore')
    try:
        alpha = minimize_scalar(f).x
    finally:
        np.seterr(**orig_settings)

    if f(alpha) > 1.0:
        raise RuntimeError("Optimizer failed to converge (error > 1.0), so "
                           "could not compute Fisher's alpha.")
    return alpha


def goods_coverage(counts):
    """Calculate Good's coverage of counts.
    Good's coverage estimator is defined as
    .. math::
       1-\\frac{F_1}{N}
    where :math:`F_1` is the number of singleton OTUs and :math:`N` is the
    total number of individuals (sum of abundances for all OTUs).
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Good's coverage estimator.
    """
    counts = _validate(counts)
    f1 = singles(counts)
    N = counts.sum()
    return 1 - (f1 / N)


def heip_e(counts):
    """Calculate Heip's evenness measure.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Heip's evenness measure.
    Notes
    -----
    The implementation here is based on the description in [1]_.
    References
    ----------
    .. [1] Heip, C. 1974. A new index measuring evenness. J. Mar. Biol. Ass.
       UK., 54, 555-557.
    """
    counts = _validate(counts)
    return ((np.exp(shannon(counts, base=np.e)) - 1) /
            (observed_otus(counts) - 1))


def kempton_taylor_q(counts, lower_quantile=0.25, upper_quantile=0.75):
    """Calculate Kempton-Taylor Q index of alpha diversity.
    Estimates the slope of the cumulative abundance curve in the interquantile
    range. By default, uses lower and upper quartiles, rounding inwards.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    lower_quantile : float, optional
        Lower bound of the interquantile range. Defaults to lower quartile.
    upper_quantile : float, optional
        Upper bound of the interquantile range. Defaults to upper quartile.
    Returns
    -------
    double
        Kempton-Taylor Q index of alpha diversity.
    Notes
    -----
    The index is defined in [1]_. The implementation here is based on the
    description given in the SDR-IV online manual [2]_.
    The implementation provided here differs slightly from the results given in
    Magurran 1998. Specifically, we have 14 in the numerator rather than 15.
    Magurran recommends counting half of the OTUs with the same # counts as the
    point where the UQ falls and the point where the LQ falls, but the
    justification for this is unclear (e.g. if there were a very large # OTUs
    that just overlapped one of the quantiles, the results would be
    considerably off). Leaving the calculation as-is for now, but consider
    changing.
    References
    ----------
    .. [1] Kempton, R. A. and Taylor, L. R. (1976) Models and statistics for
       species diversity. Nature, 262, 818-820.
    .. [2] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    n = len(counts)
    lower = int(np.ceil(n * lower_quantile))
    upper = int(n * upper_quantile)
    sorted_counts = np.sort(counts)
    return (upper - lower) / np.log(sorted_counts[upper] /
                                    sorted_counts[lower])


def margalef(counts):
    """Calculate Margalef's richness index, which is defined as:
    .. math::
       D = \\frac{(S - 1)}{\\ln N}
    where :math:`S` is the species number and :math:`N` is the
    total number of individuals (sum of abundances for all OTUs).
    Assumes log accumulation.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Margalef's richness index.
    Notes
    -----
    Based on the description in [1]_.
    References
    ----------
    .. [1] Magurran, A E 2004. Measuring biological diversity. Blackwell. pp.
       76-77.
    """
    counts = _validate(counts)
    return (observed_otus(counts) - 1) / np.log(counts.sum())


def mcintosh_d(counts):
    """Calculate McIntosh dominance index D, which is defined as:
    .. math::
       D = \\frac{N - U}{N - \\sqrt{N}}
    where :math:`N` is the total number of individuals (sum of abundances for
    all OTUs) and :math:`U` is given as:
    .. math::
        U = \\sqrt{\\sum{{n_i}^2}}
    where :math:`n_i` is the sum of abundances for all OTUs in the
    :math:`i_{th}` species.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        McIntosh dominance index D.
    See Also
    --------
    mcintosh_e
    Notes
    -----
    The index was proposed in [1]_. The implementation here is based on the
    description given in the SDR-IV online manual [2]_.
    References
    ----------
    .. [1] McIntosh, R. P. 1967 An index of diversity and the relation of
       certain concepts to diversity. Ecology 48, 1115-1126.
    .. [2] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    u = np.sqrt((counts * counts).sum())
    n = counts.sum()
    return (n - u) / (n - np.sqrt(n))


def mcintosh_e(counts):
    """Calculate McIntosh's evenness measure E.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        McIntosh evenness measure E.
    See Also
    --------
    mcintosh_d
    Notes
    -----
    The implementation here is based on the description given in [1]_, *NOT*
    the one in the SDR-IV online manual, which is wrong.
    References
    ----------
    .. [1] Heip & Engels 1974 p 560.
    """
    counts = _validate(counts)
    numerator = np.sqrt((counts * counts).sum())
    n = counts.sum()
    s = observed_otus(counts)
    denominator = np.sqrt((n - s + 1) ** 2 + s - 1)
    return numerator / denominator


def menhinick(counts):
    """Calculate Menhinick's richness index.
    Assumes square-root accumulation.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Menhinick's richness index.
    Notes
    -----
    Based on the description in [1]_.
    References
    ----------
    .. [1] Magurran, A E 2004. Measuring biological diversity. Blackwell. pp.
       76-77.
    """
    counts = _validate(counts)
    return observed_otus(counts) / np.sqrt(counts.sum())


def michaelis_menten_fit(counts, num_repeats=1, params_guess=None):
    """Calculate Michaelis-Menten fit to rarefaction curve of observed OTUs.
    The Michaelis-Menten equation is defined as
    .. math::
       S=\\frac{nS_{max}}{n+B}
    where :math:`n` is the number of individuals and :math:`S` is the number of
    OTUs. This function estimates the :math:`S_{max}` parameter.
    The fit is made to datapoints for :math:`n=1,2,...,N`, where :math:`N` is
    the total number of individuals (sum of abundances for all OTUs).
    :math:`S` is the number of OTUs represented in a random sample of :math:`n`
    individuals.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    num_repeats : int, optional
        The number of times to perform rarefaction (subsampling without
        replacement) at each value of :math:`n`.
    params_guess : tuple, optional
        Initial guess of :math:`S_{max}` and :math:`B`. If ``None``, default
        guess for :math:`S_{max}` is :math:`S` (as :math:`S_{max}` should
        be >= :math:`S`) and default guess for :math:`B` is ``round(N / 2)``.
    Returns
    -------
    S_max : double
        Estimate of the :math:`S_{max}` parameter in the Michaelis-Menten
        equation.
    See Also
    --------
    skbio.stats.subsample_counts
    Notes
    -----
    There is some controversy about how to do the fitting. The ML model given
    in [1]_ is based on the assumption that error is roughly proportional to
    magnitude of observation, reasonable for enzyme kinetics but not reasonable
    for rarefaction data. Here we just do a nonlinear curve fit for the
    parameters using least-squares.
    References
    ----------
    .. [1] Raaijmakers, J. G. W. 1987 Statistical analysis of the
       Michaelis-Menten equation. Biometrics 43, 793-803.
    """
    counts = _validate(counts)

    n_indiv = counts.sum()
    if params_guess is None:
        S_max_guess = observed_otus(counts)
        B_guess = int(round(n_indiv / 2))
        params_guess = (S_max_guess, B_guess)

    # observed # of OTUs vs # of individuals sampled, S vs n
    xvals = np.arange(1, n_indiv + 1)
    ymtx = np.empty((num_repeats, len(xvals)), dtype=int)
    for i in range(num_repeats):
        ymtx[i] = np.asarray([observed_otus(subsample_counts(counts, n))
                              for n in xvals], dtype=int)
    yvals = ymtx.mean(0)

    # Vectors of actual vals y and number of individuals n.
    def errfn(p, n, y):
        return (((p[0] * n / (p[1] + n)) - y) ** 2).sum()

    # Return S_max.
    return fmin_powell(errfn, params_guess, ftol=1e-5, args=(xvals, yvals),
                       disp=False)[0]


def observed_otus(counts):
    """Calculate the number of distinct OTUs.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    int
        Distinct OTU count.
    """
    counts = _validate(counts)
    return (counts != 0).sum()


def osd(counts):
    """Calculate observed OTUs, singles, and doubles.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    osd : tuple
        Observed OTUs, singles, and doubles.
    See Also
    --------
    observed_otus
    singles
    doubles
    Notes
    -----
    This is a convenience function used by many of the other measures that rely
    on these three measures.
    """
    counts = _validate(counts)
    return observed_otus(counts), singles(counts), doubles(counts)


def robbins(counts):
    """Calculate Robbins' estimator for the probability of unobserved outcomes.
    Robbins' estimator is defined as
    .. math::
       \\frac{F_1}{n+1}
    where :math:`F_1` is the number of singleton OTUs.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Robbins' estimate.
    Notes
    -----
    Robbins' estimator is defined in [1]_. The estimate computed here is for
    :math:`n-1` counts, i.e. the x-axis is off by 1.
    References
    ----------
    .. [1] Robbins, H. E (1968). Ann. of Stats. Vol 36, pp. 256-257.
    """
    counts = _validate(counts)
    return singles(counts) / counts.sum()


def shannon(counts, base=2):
    """Calculate Shannon entropy of counts (H), default in bits.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    base : scalar, optional
        Logarithm base to use in the calculations.
    Returns
    -------
    double
        Shannon diversity index H.
    Notes
    -----
    The implementation here is based on the description given in the SDR-IV
    online manual [1]_, except that the default logarithm base used here is 2
    instead of :math:`e`.
    References
    ----------
    .. [1] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    freqs = counts / counts.sum()
    nonzero_freqs = freqs[freqs.nonzero()]
    return -(nonzero_freqs * np.log(nonzero_freqs)).sum() / np.log(base)


def simpson(counts):
    """Calculate Simpson's index.
    Simpson's index is defined as 1 - dominance.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Simpson's index.
    See Also
    --------
    dominance
    Notes
    -----
    The implementation here is ``1 - dominance`` as described in [1]_. Other
    references (such as [2]_) define Simpson's index as ``1 / dominance``.
    References
    ----------
    .. [1] http://folk.uio.no/ohammer/past/diversity.html
    .. [2] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    return 1 - dominance(counts)


def simpson_e(counts):
    """Calculate Simpson's evenness measure E.
    Simpson's E is defined as
    .. math::
       E=\\frac{1 / D}{S_{obs}}
    where :math:`D` is dominance and :math:`S_{obs}` is the number of observed
    OTUs.
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Simpson's evenness measure E.
    See Also
    --------
    dominance
    enspie
    simpson
    Notes
    -----
    The implementation here is based on the description given in [1]_.
    References
    ----------
    .. [1] http://www.tiem.utk.edu/~gross/bioed/bealsmodules/simpsonDI.html
    """
    counts = _validate(counts)
    return enspie(counts) / observed_otus(counts)


def singles(counts):
    """Calculate number of single occurrences (singletons).
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    int
        Singleton count.
    """
    counts = _validate(counts)
    return (counts == 1).sum()


def strong(counts):
    """Calculate Strong's dominance index (Dw).
    Parameters
    ----------
    counts : 1-D array_like, int
        Vector of counts.
    Returns
    -------
    double
        Strong's dominance index (Dw).
    Notes
    -----
    Strong's dominance index is defined in [1]_. The implementation here is
    based on the description given in the SDR-IV online manual [2]_.
    References
    ----------
    .. [1] Strong, W. L., 2002 Assessing species abundance uneveness within and
       between plant communities. Community Ecology, 3, 237-246.
    .. [2] http://www.pisces-conservation.com/sdrhelp/index.html
    """
    counts = _validate(counts)
    n = counts.sum()
    s = observed_otus(counts)
    i = np.arange(1, len(counts) + 1)
    sorted_sum = np.sort(counts)[::-1].cumsum()
    return (sorted_sum / n - (i / s)).max()

#Status API Training Shop Blog About
#© 2015 GitHub, Inc. Terms Privacy Security Contact





# -*- coding: utf-8 -*-
from __future__ import division
import  matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import random
import scipy as sc
import os
import sys

import pandas
from pandas.tools import plotting
from scipy import stats
import statsmodels
from statsmodels.formula.api import ols
from numpy.random import randn


""" http://statsmodels.sourceforge.net/devel/stats.html#residual-diagnostics-and-specification-tests """


def NormStats(resids):
    
    DW = statsmodels.stats.stattools.durbin_watson(resids, axis=0) # Calculate the Durbin-Watson statistic for normality

    JB = statsmodels.stats.stattools.jarque_bera(resids, axis=0) # Calculate the Jarge-Bera test

    Omni = statsmodels.stats.stattools.omni_normtest(resids, axis=0) # Calculate the Omnibus test for normal skewnness and kurtosis

    NormAd = statsmodels.stats.diagnostic.normal_ad(x, axis=0) # Anderson-Darling test for normal distribution unknown mean and variance

    KSnorm = statsmodels.stats.diagnostic.kstest_normal(x, pvalmethod='approx') # Lillifors test for normality, Kolmogorov Smirnov test with estimated mean and variance

    Lfor = statsmodels.stats.diagnostic.lillifors(x, pvalmethod='approx') # Lillifors test for normality, Kolmogorov Smirnov test with estimated mean and variance

    return

def AutoCorrStats(x, results, lags=None, nlags=None, store=False, boxpierc=False):

    Lj = statsmodels.stats.diagnostic.acorr_ljungbox(x, lags=None, boxpierce=False) # Calculate the Ljung-Box test for no autocorrelation

    BG = statsmodels.stats.diagnostic.acorr_breush_godfrey(results, nlags=None, store=False) # Calculate the Breush Godfrey Lagrange Multiplier tests for residual autocorrelation

    return


    

def ResidStat(resid, exog_het):

    HB = statsmodels.stats.diagnostic.het_breushpagan(resid, exog_het) # The tests the hypothesis that the residual variance does not depend on the variables in x in the form

    return

def HetScedStats(resid, exog):

    HW = statsmodels.stats.diagnostic.het_white(resid, exog, retres=False) # White’s Lagrange Multiplier Test for Heteroscedasticity

    HA = statsmodels.stats.diagnostic.het_arch(resid, maxlag=None, autolag=None, store=False, regresults=False, ddof=0) # Engle’s Test for Autoregressive Conditional Heteroscedasticity (ARCH)

    return


def Linearity(res, resid, exog, olsresidual, olsresults):

    LH = statsmodels.stats.diagnostic.linear_harvey_collier(res) # Harvey Collier test for linearity. The Null hypothesis is that the regression is correctly modeled as linear.

    LR = statsmodels.stats.diagnostic.linear_rainbow(res, frac=0.5) # Rainbow test for linearity, The Null hypothesis is that the regression is correctly modelled as linear. The alternative for which the power might be large are convex, check.
 
    Llm = statsmodels.stats.diagnostic.linear_lm(resid, exog, func=None) # Lagrange multiplier test for linearity against functional alternative

    Bcusum = statsmodels.stats.diagnostic.breaks_cusumolsresid(olsresidual, ddof=0) # cusum test for parameter stability based on ols residuals

    BH = statsmodels.stats.diagnostic.breaks_hansen(olsresults) # test for model stability, breaks in parameters for ols, Hansen 1992

    Rols = statsmodels.stats.diagnostic.recursive_olsresiduals(olsresults, skip=None, lamda=0.0, alpha=0.95) # calculate recursive ols with residuals and cusum test statistic


def Outliers(results):
    
    #class statsmodels.stats.outliers_influence.OLSInfluence(results)
    OutInf = statsmodels.stats.outliers_influence.OLSInfluence(results)

    return
    
""" Incomplete
# HGQ = statsmodels.stats.diagnostic.HetGoldfeldQuandt # function is not complete in statsmodels documentation
# ComCox = class statsmodels.stats.diagnostic.CompareCox # Cox Test for non-nested models
"""







"""Module for conducting standard macroecological plots and analyses"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from numpy import log10
import pandas
import numpy

def AIC(k, L):
    """Computes the Akaike Information Criterion.

    Keyword arguments:
    L  --  log likelihood value of given distribution.
    k  --  number of fitted parameters.

    """
    AIC = 2 * k - 2 * L
    return AIC


def AICc(k, L, n):
    """Computes the corrected Akaike Information Criterion.

    Keyword arguments:
    L  --  log likelihood value of given distribution.
    k  --  number of fitted parameters.
    n  --  number of observations.

    """
    AICc = 2 * k - 2 * L + 2 * k * (k + 1) / (n - k - 1)
    return AICc

def aic_weight(AICc_list, n, cutoff = 4):
    """Computes Akaike weight for one model relative to others

    Based on information from Burnham and Anderson (2002).

    Keyword arguments:
    n           --  number of observations.
    cutoff      --  minimum number of observations required to generate a weight.
    AICc_list   --  list of AICc values for each model

    """
    if n < cutoff:
        AICc_weights = None

    else:
        AICc_min = min(AICc_list) # Minimum AICc value for the entire list
        relative_likelihoods = []

        for AICc in AICc_list:
            delta_AICc = AICc - AICc_min
            relative_likelihood = np.exp(-(delta_AICc)/2)
            relative_likelihoods.append(relative_likelihood)

        relative_likelihoods = np.array(relative_likelihoods)

        AICc_weights = relative_likelihoods / sum(relative_likelihoods)

        return(AICc_weights)

def get_pred_iterative(cdf_obs, dist, *pars):
    """Function to get predicted abundances (reverse-sorted) for distributions with no analytical ppf."""
    cdf_obs = np.sort(cdf_obs)
    abundance  = list(np.empty([len(cdf_obs)]))
    j = 0
    cdf_cum = 0
    i = 1
    while j < len(cdf_obs):
        cdf_cum += dist.pmf(i, *pars)
        while cdf_cum >= cdf_obs[j]:
            abundance[j] = i
            j += 1
            if j == len(cdf_obs):
                abundance.reverse()
                return np.array(abundance)
        i += 1

def get_rad_from_cdf(dist, S, *args):
    """Return a predicted rank-abundance distribution from a theoretical CDF

    Keyword arguments:
    dist -- a distribution class
    S -- the number of species for which the RAD should be predicted. Should
    match the number of species in the community if comparing to empirical data.
    args -- arguments for dist

    Finds the predicted rank-abundance distribution that results from a
    theoretical cumulative distribution function, by rounding the value of the
    cdf evaluated at 1 / S * (Rank - 0.5) to the nearest integer

    """
    emp_cdf = [(S - i + 0.5) / S for i in range(1, S + 1)]
    try: rad = int(np.round(dist.ppf(emp_cdf, *args)))
    except: rad = get_pred_iterative(emp_cdf, dist, *args)
    return np.array(rad)

def get_emp_cdf(dat):
    """Compute the empirical cdf given a list or an array"""
    dat = np.array(dat)
    emp_cdf = []
    for point in dat:
        point_cdf = len(dat[dat <= point]) / len(dat)
        emp_cdf.append(point_cdf)
    return np.array(emp_cdf)

def hist_pmf(xs, pmf, bins):
    """Create a histogram based on a probability mass function

    Creates a histogram by combining the pmf values inside a series of bins

    Args:
        xs: Array-like list of x values that the pmf values come from
        pmf: Array-like list of pmf values associate with the values of x
        bins: Array-like list of bin edges

    Returns:
        hist: Array of values of the histogram
        bin_edges: Array of value of the bin edges

    """
    xs = np.array(xs)
    pmf = np.array(pmf)
    bins = np.array(bins)

    hist = []
    for lower_edge_index in range(len(bins) - 1):
        if lower_edge_index + 1 == len(bins):
            hist.append(sum(pmf[(xs >= bins[lower_edge_index]) &
                               (xs <= bins[lower_edge_index + 1])]))
        else:
            hist.append(sum(pmf[(xs >= bins[lower_edge_index]) &
                               (xs < bins[lower_edge_index + 1])]))
    hist = np.array(hist)
    return (hist, bins)

def plot_rad(Ns):
    """Plot a rank-abundance distribution based on a vector of abundances"""
    Ns.sort(reverse=True)
    rank = range(1, len(Ns) + 1)
    plt.plot(rank, Ns, 'bo-')
    plt.xlabel('Rank')
    plt.ylabel('Abundance')

def get_rad_data(Ns):
    """Provide ranks and relative abundances for a vector of abundances"""
    Ns = np.array(Ns)
    Ns_sorted = -1 * np.sort(-1 * Ns)
    relab_sorted = Ns_sorted / sum(Ns_sorted)
    rank = range(1, len(Ns) + 1)
    return (rank, relab_sorted)

def preston_sad(abund_vector, b=None, normalized = 'no'):
    """Plot histogram of species abundances on a log2 scale"""
    if b == None:
        q = np.exp2(list(range(0, 25)))
        b = q [(q <= max(abund_vector)*2)]

    if normalized == 'no':
        hist_ab = np.histogram(abund_vector, bins = b)
    if normalized == 'yes':
        hist_ab_norm = np.histogram(abund_vector, bins = b)
        hist_ab_norm1 = hist_ab_norm[0]/(b[0:len(hist_ab_norm[0])])
        hist_ab_norm2 = hist_ab_norm[1][0:len(hist_ab_norm[0])]
        hist_ab = (hist_ab_norm1, hist_ab_norm2)
    return hist_ab

def plot_SARs(list_of_A_and_S):
    """Plot multiple SARs on a single plot.

    Input: a list of lists, each sublist contains one vector for S and one vector for A.
    Output: a graph with SARs plotted on log-log scale, with colors spanning the spectrum.

    """
    N = len(list_of_A_and_S)
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    for i in range(len(list_of_A_and_S)):
        sublist = list_of_A_and_S[i]
        plt.loglog(sublist[0], sublist[1], color = RGB_tuples[i])
    plt.hold(False)
    plt.xlabel('Area')
    plt.ylabel('Richness')

def count_pts_within_radius(x, y, radius, logscale=0):
    """Count the number of points within a fixed radius in 2D space"""
    #TODO: see if we can improve performance using KDTree.query_ball_point
    #http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query_ball_point.html
    #instead of doing the subset based on the circle
    raw_data = np.array([x, y])
    x = np.array(x)
    y = np.array(y)
    raw_data = raw_data.transpose()

    # Get unique data points by adding each pair of points to a set
    unique_points = set()
    for xval, yval in raw_data:
        unique_points.add((xval, yval))

    count_data = []
    for a, b in unique_points:
        if logscale == 1:
            num_neighbors = len(x[((log10(x) - log10(a)) ** 2 +
                                   (log10(y) - log10(b)) ** 2) <= log10(radius) ** 2])
        else:
            num_neighbors = len(x[((x - a) ** 2 + (y - b) ** 2) <= radius ** 2])
        count_data.append((a, b, num_neighbors))
    return count_data

def plot_color_by_pt_dens(x, y, radius, loglog=0, plot_obj=None):
    """Plot bivariate relationships with large n using color for point density

    Inputs:
    x & y -- variables to be plotted
    radius -- the linear distance within which to count points as neighbors
    loglog -- a flag to indicate the use of a loglog plot (loglog = 1)

    The color of each point in the plot is determined by the logarithm (base 10)
    of the number of points that occur with a given radius of the focal point,
    with hotter colors indicating more points. The number of neighboring points
    is determined in linear space regardless of whether a loglog plot is
    presented.

    """
    plot_data = count_pts_within_radius(x, y, radius, loglog)
    sorted_plot_data = np.array(sorted(plot_data, key=lambda point: point[2]))

    if plot_obj == None:
        plot_obj = plt.axes()

    if loglog == 1:
        plot_obj.set_xscale('log')
        plot_obj.set_yscale('log')
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                         c = np.sqrt(sorted_plot_data[:, 2]), edgecolors='none')
        plot_obj.set_xlim(min(x) * 0.5, max(x) * 2)
        plot_obj.set_ylim(min(y) * 0.5, max(y) * 2)
    else:
        plot_obj.scatter(sorted_plot_data[:, 0], sorted_plot_data[:, 1],
                    c = log10(sorted_plot_data[:, 2]), edgecolors='none')
    return plot_obj

def e_var(abundance_data):
    """Calculate Smith and Wilson's (1996; Oikos 76:70-82) evenness index (Evar)

    Input:
    abundance_data = list of abundance fo all species in a community

    """
    S = len(abundance_data)
    ln_nj_over_S=[]
    for i in range(0, S):
        v1 = (np.log(abundance_data[i]))/S
        ln_nj_over_S.append(v1)

    ln_ni_minus_above=[]
    for i in range(0, S):
        v2 = ((np.log(abundance_data[i])) - sum(ln_nj_over_S)) ** 2
        ln_ni_minus_above.append(v2)

    return(1 - ((2 / np.pi) * np.arctan(sum(ln_ni_minus_above) / S)))

def obs_pred_rsquare(obs, pred):
    """Determines the prop of variability in a data set accounted for by a model

    In other words, this determines the proportion of variation explained by
    the 1:1 line in an observed-predicted plot.

    """
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

def obs_pred_mse(obs, pred):
    """Calculate the mean squared error of the prediction given observation."""

    return sum((obs - pred) ** 2) / len(obs)

def comp_ed (spdata1,abdata1,spdata2,abdata2):
    """Calculate the compositional Euclidean Distance between two sites

    Ref: Thibault KM, White EP, Ernest SKM. 2004. Temporal dynamics in the
    structure and composition of a desert rodent community. Ecology. 85:2649-2655.

    """
    abdata1 = (abdata1 * 1.0) / sum(abdata1)
    abdata2 = (abdata2 * 1.0) / sum(abdata2)
    intersect12 = set(spdata1).intersection(spdata2)
    setdiff12 = np.setdiff1d(spdata1,spdata2)
    setdiff21 = np.setdiff1d(spdata2,spdata1)
    relab1 = np.concatenate(((abdata1[np.setmember1d(spdata1,list(intersect12)) == 1]),
                             abdata1[np.setmember1d(spdata1,setdiff12)],
                             np.zeros(len(setdiff21))))
    relab2 = np.concatenate((abdata2[np.setmember1d(spdata2,list(intersect12)) == 1],
                              np.zeros(len(setdiff12)),
                              abdata2[np.setmember1d(spdata2,setdiff21)]))
    return np.sqrt(sum((relab1 - relab2) ** 2))

def calc_comp_eds(ifile, fout, cutoff=4):
    """Calculate Euclidean distances in species composition across sites.

    Determines the Euclidean distances among all possible pairs of sites and
    saves the results to a file

    Inputs:
    ifile -- ifile = np.genfromtxt(input_filename, dtype = "S15,S15,i8",
                   names = ['site','species','ab'], delimiter = ",")
    fout -- fout = csv.writer(open(output_filename,'ab'))

    """
    #TODO - Remove reliance on on names of columns in input
    #       Possibly move to 3 1-D arrays for input rather than the 2-D with 3 columns
    #       Return result rather than writing to file

    usites = np.sort(list(set(ifile["site"])))

    for i in range (0, len(usites)-1):
        spdata1 = ifile["species"][ifile["site"] == usites[i]]
        abdata1 = ifile["ab"][ifile["site"] == usites[i]]

        for a in range (i+1,len(usites)):
            spdata2 = ifile["species"][ifile["site"] == usites[a]]
            abdata2 = ifile["ab"][ifile["site"] == usites[a]]

            if len(spdata1) > cutoff and len(spdata2) > cutoff:
                ed = comp_ed (spdata1,abdata1,spdata2,abdata2)
                results = np.column_stack((usites[i], usites[a], ed))
                fout.writerows(results)

def combined_spID(*species_identifiers):
    """Return a single column unique species identifier

    Creates a unique species identifier based on one or more columns of a
    data frame that represent the unique species ID.

    Args:
        species_identifiers: A tuple containing one or pieces of a unique
            species identifier or lists of these pieces.

    Returns:
        A single unique species identifier or a list of single identifiers

    """

    # Make standard input data types capable of element wise summation
    input_type = type(species_identifiers[0])
    assert input_type in [list, tuple, str, pandas.core.series.Series, numpy.ndarray]
    if input_type is not str:
        species_identifiers = [pandas.Series(identifier) for identifier in species_identifiers]

    single_identifier = species_identifiers[0]
    if len(species_identifiers) > 1:
        for identifier in species_identifiers[1:]:
            single_identifier += identifier
    if input_type == numpy.ndarray:
        single_identifier = numpy.array(single_identifier)
    else:
        single_identifier = input_type(single_identifier)
    return single_identifier

def richness_in_group(composition_data, group_cols, spid_cols):
    """Determine the number of species in a grouping (e.g., at each site)

    Counts the number of species grouped at one or more levels. For example,
    the number of species occuring at each of a series of sites or in each of
    a series of years.

    If a combination of grouping variables is not present in the data, then no
    values will be returned for that combination. In other words, if these
    missing combinations should be treated as zeros, this will need to be
    handled elsewhere.

    Args:
        composition_data: A Pandas data frame with one or more columns with
            information on species identity and one or more columns with
            information on the groups, e.g., years or sites.
        group_cols: A list of strings of the names othe columns in
            composition_data that hold the grouping fields.
        spid_cols: A list of strings of the names of the columns in
            composition_data that hold the data on species ID. This could be a
            single column with a unique ID or two columns containing the latin binomial.

    Returns:
        A data frame with the grouping fields and the species richness

    """
    spid_series = [composition_data[spid_col] for spid_col in spid_cols]
    single_spid = combined_spID(*spid_series)
    composition_data['_spid'] = single_spid
    richness = composition_data.groupby(group_cols)._spid.nunique()
    richness = richness.reset_index()
    richness.columns = group_cols + ['richness']
    del composition_data['_spid']
    return richness

def abundance_in_group(composition_data, group_cols, abund_col=None):
    """Determine the number of individuals in a grouping (e.g., at each site)

    Counts the number of individuals grouped at one or more levels. For example,
    the number of species occuring at each of a series of sites or in each of
    a series of genus-species combinations.

    If a combination of grouping variables is not present in the data, then no
    values will be returned for that combination. In other words, if these
    missing combinations should be treated as zeros, this will need to be
    handled elsewhere.

    Args:
        composition_data: A Pandas data frame with one or more columns with
            information on species identity and one or more columns with
            information on the groups, e.g., years or sites, and a column
            containing the abundance value.
        group_cols: A list of strings of the names othe columns in
            composition_data that hold the grouping fields.
        abund_col: A column containing abundance data. If this column is not
            provided then it is assumed that the abundances are to be obtained
            by counting the number of rows in the group (e.g., there is one
            sample per individual in many ecological datasets)

    Returns:
        A data frame with the grouping fields and the species richness

    """
    if abund_col:
        abundance = composition_data[group_cols + abund_col].groupby(group_cols).sum()
    else:
        abundance = composition_data[group_cols].groupby(group_cols).size()
    abundance = pandas.DataFrame(abundance)
    abundance.columns = ['abundance']
    abundance = abundance.reset_index()
    return abundance
    
    
    
    
    

