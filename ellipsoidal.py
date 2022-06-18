import numpy as np
import numpy.linalg

from scipy.integrate import quad

import itertools
from preference_classes import Agent, Query, generate_items


def test():

    # mu = initial mean,
    # sigma = initial covariance

    num_features = 4
    num_items = 20
    seed = 0

    agent = Agent.random(num_features)

    items = generate_items(num_features, num_items, seed=seed)

    # initialize uncertainty set (prior?)
    a = -1.0
    b = 1.0

    mu, cov = initialize_uniform_prior(a, b, num_features)

    # simulate elicitation...
    for i in range(50):

        # get next query
        q = get_next_query_exhaustive(mu, cov, items)

        # simulate agent response...
        agent.answer_query(q)
        new_query = agent.answered_queries[-1]

        # update agent model (mu and sigma)
        mu, cov = update_bayes_approximation(new_query, mu, cov)
        print(f"mu={mu}")
        print(f"true u={agent.u_true}")



def initialize_uniform_prior(a, b, n):
    """return mean and covariance of n independent uniform distributions on [a, b]"""
    mean = (a + b) / 2.0
    var = ((b - a) ** 2) / 12.0

    mu = np.ones(n) * mean
    cov = np.eye(n) * var

    return mu, cov


def get_next_query_exhaustive(mu, sig, items):
    """
    find the next query by solving minimization problem (8) in Vielma et al.

    solve this problem by exhaustively checking each possible query (all pairs of items)

    if multiple queries are optimal, return the first one found

    the next query is the pair of items (x, y) one that minimizes g(a, b) where
    a = (x - y) \cdot mu
    b = || (cov^{1/2})^T (x - y) ||
    """

    sigma_sqrt_t = np.linalg.cholesky(sig).T

    assert mu.shape == items[0].features.shape

    best_g = 999.0
    best_query = None
    for item_a, item_b in itertools.combinations(items, 2):
        z = item_a.features - item_b.features
        a = np.dot(z, mu)
        b = np.linalg.norm(np.dot(sigma_sqrt_t, z), 2) ** 2.0  # this is squared, since we take the sqrt of b inside g_func
        g = g_func(a, b)
        if g < best_g:
            best_query = Query(item_a, item_b)
            best_g = g

    assert best_query is not None

    return best_query






#  ------ FROM ellipsoidal.jl: getquestion -------
#     n = size(Σ, 1)
#     m = JuMP.Model(solver=mip_solver)
#
#     # define variables for linearization
#     JuMP. @ variable(m, 0 <= x[1:n] <= 1, Int)
#     JuMP. @ variable(m, 0 <= y[1:n] <= 1, Int)
#     # x ≠ y
#     JuMP. @ constraint(m, linquad(m, (x - y)⋅(x - y)) >= 1)
#     # v = x-y, β ∼ 𝒩(μ,Σ), v⋅β ∼ 𝒩(μᵥ,σ²), μᵥ = μ⋅v, σ² = v'*Σ*v
#     JuMP. @ variable(m, μᵥ)
#     JuMP. @ constraint(m, μᵥ == μ⋅(x - y) )
#     JuMP. @ variable(m, σ² >= 0)
#     JuMP. @ constraint(m, σ² == linquad(m, (x - y)⋅(Σ * (x - y))))
#     # (x-y)'*Σ*(x-y) <= eigmax(Σ) ||x-y||₂ <= eigmax(Σ)*n
#     σ̅² = eigmax(Σ) * n
#     # (x-y)'*Σ*(x-y) >= eigmin(Σ) ||x-y||₂ >= eigmin(Σ)   ( x ≠ y )
#     σ̲² = eigmin(Σ)
#     μ̅ᵥ = norm(μ, 1)
#
#     μᵥnpoints = 2 ^ k - 1
#     μᵥpoints = []
#     if μ̅ᵥ > 1e-6
#         μᵥpoints = 0:μ̅ᵥ / μᵥnpoints: μ̅ᵥ + (μ̅ᵥ / μᵥnpoints) / 2
#     else
#         μᵥpoints = 0:1e-6: 1e-6  # 0:0
#     end
#     σ²points = []
#     σ²range = σ̅² - σ̲²
#     σ²npoints = 2 ^ k - 1
#     if σ²range > 1e-6
#     σ²points = σ̲²:σ²range / σ²npoints: σ̅²+(σ²range / σ²npoints) / 2
#
# else
# σ²points = σ̲²:1e-6: σ̲²+1e-6  # σ̲²: σ̲²
# end
# pwl = PiecewiseLinearOpt.BivariatePWLFunction(μᵥpoints, σ²points, (μᵥ, σ²) -> variancefuction(μᵥ, sqrt(
#     σ²)); pattern =:UnionJack)
#
# obj = PiecewiseLinearOpt.piecewiselinear(m, μᵥ, σ², pwl;
# method =:Logarithmic)
# JuMP. @ objective(m, Min, obj)


# # Returns the "confidece" confidence level credibility ellipsoid
# # for a gaussian with mean mu and covariate matrix sigma
# #
# # Parameters:
# #
# #   - mu = mean of original prior distribution of beta
# #   - sigma = covariance matrix of original prior distribution of beta
# #	- confidence = confidence level for the ellipsoid
# #
# # Returns
# #
# #	- ellipsoid (beta - c)'*Q^(-1)*(beta - c)
#
# function initializeEllipsoid(mu,sigma,precomp)
#
#
# 	Q = Array{Float64,2}[]
# 	push!(Q,sigma*precomp["rhs"])
# 	c = Array{Float64,1}[]
# 	push!(c,mu)
#
# 	return Q,c,Array{Float64,1}[],Float64[]
# end


# # Assumes the uncertainty set is a single ellipsoid and
# # hence the ellipsoidal approximation is the ellipsoid itself.
# # If Q[1] is numerically non-symmetric it tries to fix it.
# #
# # Parameters:
# #
# # - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# # - questions, answers = history of questions and answers
# # - precomp = precompuded data
# # - dimension = the dimension of beta
# # - parameters = list of parameters
# #
# # Returns:
# #
# # - ellipsoidal approximation (beta - c[1])'*Q[1]^(-1)*(beta - c[1])
# # - status, which is always "Normal"
#
# function estimateEllipsoid(Q,c,A,b,questions,answers,precomp,dimension,parameters)
# 	if !issymmetric(Q[1])
# 		for i in 1:size(Q[1],1)
# 			for j in i+1:size(Q[1],1)
# 				if abs(Q[1][i,j]-Q[1][j,i]) > 1e-5
# 					return c[1], Q[1], "UnsymmetricMatrix"
# 				else
# 					Q[1][i,j]=Q[1][j,i]
# 				end
# 			end
# 		end
# 	end
#
# 	L,V = eig(Q[1])
# 	order = sortperm(L,rev=true)
#
# 	return c[1], Q[1],V[:,order[1]], "Normal"
# end


# # Generates precomputed data
# #
# # Parameters:
# #
# # - mu = mean of prior
# # - sigma = covariance  of prior
# # - confidence = level for confidence ellipsoid/polyhedron
# #
# # Returns
# #
# # - precomputed data
# function precompute(mu,sigma,confidence)
# 	precomp = Dict{AbstractString,Any}()
#
# 	distro = Chisq(length(mu))
# 	precomp["rhs"] = quantile(distro, confidence)
# 	precomp["alpha"] = getAlpha(mu)
#
# 	precomp
# end


def update_bayes_approximation(next_answered_query, old_center, cov):
    """
	 this function is the test loop for Vielma et al. Ellipsoidal Methods.

	 the function structure is almost identical to function update.jl:update_bayes_approximation. This implementation
	 includes all all functionality from other subfunctions (each subfunction is mentioned in the comments below).
	"""

    # get the more-preferred and less-preferred items from the last query
    if next_answered_query.response not in [-1, 1]:
        raise Exception("query response must be a strict preference (1 or -1)")

    if next_answered_query.response == 1:
        x = next_answered_query.item_A.features
        y = next_answered_query.item_B.features
    else:
        x = next_answered_query.item_B.features
        y = next_answered_query.item_A.features

    # ------------------------------------------------------------------
    # below is adapted from function ellipsoidal.jl:momentmatchingupdate

    # old_cov = sigma_sqrt * sigma_sqrt^T
    sigma_sqrt = np.linalg.cholesky(cov)

    v = x - y
    n = len(v)

    # make v a column vector
    v_vec = v.reshape(n, 1)

    # reshape mu to be a column vector as well
    mu_vec = old_center.reshape(n, 1)

    # W is an orthogonal matrix, s.t.:
    # W[:, 1] = (1 / sig_xy) * sigma_sqrt^T * v, where...
    # sig_xy = || sigma_sqrt^T * v || = sqrt( v^T * sigma * v) = r, where...
    # r^2 = v^T * sigma * v  (by definition)
    #
    # in this QR decomp, r may be negative.  W is just a column vector:
    # W, r = np.linalg.qr(np.dot(sigma_sqrt.T, v_vec))
    W, r_vec = np.linalg.qr(np.dot(sigma_sqrt.T, v_vec), mode="complete")

    r = r_vec[0, 0]
    # TODO: should W_minus be used anywhere? it isn't in the original julia code.
    W_minus = np.sign(float(r)) * W
    r_plus = abs(float(r))

    mu_v = float(np.inner(v_vec.T, mu_vec.T))

    sig = r_plus

    # --------------------------------------------------------
    # below is adapted from function ellipsoidal.jl:qgk_update

    def c_integrand(x):
        return (
            (1.0 / (1.0 + np.exp(-mu_v - sig * x)))
            * np.exp(-(x ** 2) / 2)
            / np.sqrt(2 * np.pi)
        )

    def mu_z_integrand(x):
        return (
            x
            * (1.0 / (1.0 + np.exp(-mu_v - sig * x)))
            * np.exp(-(x ** 2) / 2)
            / np.sqrt(2 * np.pi)
        )

    def var_z_integrand(x):
        return (
            x
            * x
            * (1.0 / (1.0 + np.exp(-mu_v - sig * x)))
            * np.exp(-(x ** 2) / 2)
            / np.sqrt(2 * np.pi)
        )

    c, _ = quad(c_integrand, -np.inf, np.inf)
    mu_z_int, _ = quad(mu_z_integrand, -np.inf, np.inf)
    mu_z = mu_z_int / c
    var_z_int, _ = quad(var_z_integrand, -np.inf, np.inf)
    var_z = var_z_int / c - (mu_z ** 2)

    mu_new = old_center + (1.0 / sig) * np.dot(cov, v) * mu_z

    temp = np.dot(sigma_sqrt, W)
    mat = np.eye(n)
    mat[0, 0] = var_z
    cov_new = np.dot(np.dot(temp, mat), temp.T)

    return mu_new, cov_new

INF = np.inf
def g_func(mu_v, sig, r=2.0):
    """directly adapted from function ellipsoidal.jl:qgk_deff"""

    # this is not in the original code, but it is necessary
    sig = np.sqrt(sig)

    def c_integrand(x):
        return (
            (1.0 / (1.0 + np.exp(-mu_v - sig * x)))
            # (np.exp(mu_v + sig * x) / (1.0 + np.exp(mu_v + sig * x)))
            * np.exp(-(x ** 2) / 2.0)
            / np.sqrt(2 * np.pi)
        )

    def mu_z1_integrand(x):
        return (
            x * (1.0 / (1.0 + np.exp(-mu_v - sig * x))) * np.exp(-(x ** 2) / 2.0)
            # x * (np.exp(mu_v + sig * x) / (1.0 + np.exp(mu_v + sig * x))) * np.exp(-(x ** 2) / 2.0)
        ) / np.sqrt(2 * np.pi)

    def var_z1_integrand(x):
        return (
            x * x * (1.0 / (1.0 + np.exp(-mu_v - sig * x))) * np.exp(-(x ** 2) / 2.0)
            # x * x * (np.exp(mu_v + sig * x) / (1.0 + np.exp(mu_v + sig * x))) * np.exp(-(x ** 2) / 2.0)
        ) / np.sqrt(2 * np.pi)

    def mu_z2_integrand(x):
        return (
            (
                x
                * (1.0 - (1.0 / (1.0 + np.exp(-mu_v - sig * x))))
                # * (1.0 - (np.exp(mu_v + sig * x) / (1.0 + np.exp(mu_v + sig * x))))
                * np.exp(-(x ** 2) / 2.0)
            )
        ) / np.sqrt(2 * np.pi)

    def var_z2_integrand(x):
        return (
            (
                x
                * x
                * (1.0 - (1.0 / (1.0 + np.exp(-mu_v - sig * x))))
                # * (1.0 - (np.exp(mu_v + sig * x) / (1.0 + np.exp(mu_v + sig * x))))
                * np.exp(-(x ** 2) / 2.0)
            )
        ) / np.sqrt(2 * np.pi)

    c, _ = quad(c_integrand, -INF, INF)

    if c < 1e-6:
        c = 0.0
        var_z1 = 1.0
        print(f"trivial c={c} for mu={mu_v}, sig={sig}")
    else:
        mu_z1_int, _ = quad(mu_z1_integrand, -INF, INF)
        mu_z1 = mu_z1_int / c
        var_z1_int, _ = quad(var_z1_integrand, -INF, INF)
        var_z1 = var_z1_int / c - (mu_z1 ** 2)

    if 1.0 - c < 1e-6:
        c = 1.0
        var_z2 = 1.0
        print(f"trivial 1-c (c={c}) for mu={mu_v}, sig={sig}")

    else:
        mu_z2_int, _ = quad(mu_z2_integrand, -INF, INF)
        mu_z2 = mu_z2_int / (1.0 - c)
        var_z2_int, _ = quad(var_z2_integrand, -INF, INF)
        var_z2 = var_z2_int / (1.0 - c) - (mu_z2 ** 2)

    return c * (var_z1 ** (1.0 / r)) + (1.0 - c) * (var_z2 ** (1.0 / r))



# import numpy as np
# import numpy.linalg
# from scipy.integrate import quad
#
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np
#
#
# fig = plt.figure()
# ax = fig.gca(projection="3d")
#
# # Make data.
# M = np.arange(-5, 5, 0.25, dtype=np.float128)
# V = np.arange(1, 10, 0.25, dtype=np.float128)
# X, Y = np.meshgrid(M, V)
#
# r = 2.0
# g = lambda x, y: g_func(x, y, r=r)
# g_func_vectorized = np.vectorize(g)
# Z = g_func_vectorized(X, Y)
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#
# ax.view_init(elev=19.0, azim=114.0)
# # Customize the z axis.
# ax.set_zlim(0.7, 1.0)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
#
# plt.show()
#
# # create some dimension lists
# X1 = np.arange(-5, 5, 0.25, dtype=np.float128)
# X2 = np.arange(-10, 10, 0.25, dtype=np.float128)
# X3 = np.arange(-1, 1, 0.25, dtype=np.float128)
#
# n = 3
#
# # this may need to be a tuple, but could be a list(?)
# spacing_tuple = [X1, X2, X3]
#
# meshgrid_tuple = np.meshgrid(*spacing_tuple)
#
# # this function takes a variable number of args (args is a tuple)
# def my_func_n(*args):
#     # print(args)
#     return sum(args)
#
# # vectorize my function (which takes a variable number of args)
# my_func_vectorized = np.vectorize(my_func_n)
#
# # pass in the meshgrid list to my vectorized function
# Z = my_func_vectorized(*meshgrid_tuple)


if __name__=="__main__":
    test()