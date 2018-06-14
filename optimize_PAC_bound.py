import cvxpy as cvx
import numpy as np
import time

# Optimize PAC bound using Relative Entropy Programming
def optimize_PAC_bound(costs_precomputed, p0, delta):
    
    # Number of actions
    L = len(p0)
    
    # Number of environments
    m = np.shape(costs_precomputed)[0]
    
    # Discretize lambdas
    lambdas = np.linspace(0,1,100)
    
    # Initialize vectors for storing optimal solutions
    taus = np.zeros(len(lambdas))
    ps = len(lambdas)*[None]

    for k in range(len(lambdas)):

        lambda0 = lambdas[k]

        # Create cost function variable
        tau = cvx.Variable()

        # Create variable for probability vector
        p = cvx.Variable(L)

        cost_empirical = (1/m)*cvx.sum(costs_precomputed*p)

        # Constraints
        constraints = [lambda0**2 >= (cvx.sum(cvx.kl_div(p, p0)) + np.log(2*np.sqrt(m)/delta))/(2*m), lambda0 == (tau - cost_empirical), p >= 0, cvx.sum(p) == 1]
        
        prob = cvx.Problem(cvx.Minimize(tau), constraints)

        # Solve problem
        opt = prob.solve(verbose=False, solver=cvx.SCS) # , max_iters=3000)

        # Store optimal value and optimizer
        if (opt > 1.0):
            taus[k] = 1.0
            ps[k] = p.value
        else:        
            taus[k] = opt
            ps[k] = p.value
    
    # Find minimizer
    min_ind = np.argmin(taus)
    tau_opt = taus[min_ind]
    p_opt = ps[min_ind]
    
    return tau_opt, p_opt, taus

# Compute kl inverse using Relative Entropy Programming
def kl_inverse(q, c):
    
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q,1-q])

    constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli,p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    opt = prob.solve(verbose=False, solver=cvx.SCS) # solver=cvx.ECOS
    
    return p_bernoulli.value[0] 
    



