import cvxpy as cvx
import numpy as np
import time

# Compute KL inverse using Relative Entropy Programming
def kl_inverse(q, c):
    
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q,1-q])

    constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli,p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    opt = prob.solve(verbose=False, solver=solver=cvx.SCS) # solver=cvx.ECOS
    
    return p_bernoulli.value[0] 
