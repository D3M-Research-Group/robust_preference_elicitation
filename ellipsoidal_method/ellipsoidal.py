# implementation of Ellipsoidal Methods for Adaptive Choice-Based Conjoint Analysis (Denis Suare & JP Vielma)

# this is currently just a skeleton

import numpy as np
from numpy import linalg

def initial_prior(num_features):
    """create a simple initial prior"""

    # zero mean
    mu = np.zeros(num_features)

    # identity covariange
    sigma = np.eye(num_features)

    return mu, sigma


def update_posterior(mu_prior, sigma_prior, answered_query):
    """
    take a prior on mu, sigma, and an answered query, and return the updated posterior for mu and sigma

    see Prop 1 of Saure & Vielma
    """

    # TODO: check answered_query
    # TODO: check mu_prior
    # TODO: check sigma_prior

    # get cholesky decomp for prior sigma
    sigma_sqrt = linalg.cholesky(sigma_prior)

    # get query vector
    if answered_query.response == 1:
        v = answered_query.z
    elif answered_query.response == -1:
        v = - answered_query.z
    else:
        raise Exception("answered_query must have response 1 or -1")

    # number of features
    n = len(answered_query.item_A.features)

    # create orthogonal matrix w, using the QR-decomposition of sigma
    q, r = np.linalg.qr(sigma_sqrt)

    # W is orthogonal matrix, W[:,1] = (1 / r) * Σ½'*v, σ² = v'*Σ*v = r²
    # Σ½ * W[:,1] = (1 / r) * Σ * v
    # v' * Σ½ * W = [(( 1 / r) * v' * Σ * v)  0 ⋯ 0 ] = [ σ * sign(r)  0 ⋯ 0 ]
    W,r = qr(Σ½'*v[:,:],;thin=false)
    # W is orthogonal matrix, W[:,1] = (1 / r) * Σ½'*v, σ² = v'*Σ*v = r²
    # r₊ = abs(r) ≥ 0, σ = r₊, W₋ = sign(r) * W
    # (Σ½ * W₋)[:,1] = Σ½ * W₋[:,1] = (1 / σ) * Σ * v
    # v' * Σ½ * W₋ = [((1 / σ) * v' * Σ * v)  0 ⋯ 0 ] = [ σ  0 ⋯ 0 ]
    W₋ = sign(r[]) * W
    r₊ = abs(r[])

    μᵥ = v⋅μ
    σ  = r₊
    μz, σ²z = updatefunction(μᵥ,σ)
    temp =  Σ½ * W

    return μ + (1 / σ) * Σ * v * μz, temp * [σ²z zeros(1,n-1); zeros(n-1,1) eye(n-1)] *temp'

    return mu, sigma


def calc_query_value(query, mu, sigma):
    """calculate the value g() of a query (proportional to the D-error of the posterior, after updating)"""

    pass