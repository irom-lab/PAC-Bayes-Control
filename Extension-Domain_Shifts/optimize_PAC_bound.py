from utils import *

#Opt pac bound
def optimize_PAC_bound_NPB(costs_precomputed, p0, delta, N, params):

    # Number of actions
    L = len(p0)

    # Discretize lambdas
    lambdas = np.linspace(0,1,100)

    # Initialize vectors for storing optimal solutions
    taus = np.zeros(len(lambdas))
    ps = len(lambdas)*[None]
    E_s_Z = 2*np.sqrt(N)

    for k in range(len(lambdas)):

        lambda0 = lambdas[k]

        # Create cost function variable
        tau = cvx.Variable()

        # Create variable for probability vector
        p = cvx.Variable(L)

        cost_empirical = (1/N)*cvx.sum(costs_precomputed*p)

        # Constraints
        constraints = [lambda0**2 >= (cvx.sum(cvx.kl_div(p, p0)) + np.log(E_s_Z/delta))/(2*N), lambda0 == (tau - cost_empirical), p >= 0, cvx.sum(p) == 1]

        prob = cvx.Problem(cvx.Minimize(tau), constraints)

        # Solve problem
        opt = prob.solve(verbose=False, solver=params['solver']) # , max_iters=3000)
        # Store optimal value and optimizer
        if (opt > 1.0):
            taus[k] = 1.0
        else:
            taus[k] = opt
        ps[k] = p.value

    # Find minimizer
    min_ind = np.argmin(taus)
    tau_opt = taus[min_ind]
    p_opt = ps[min_ind]
    Cp = (taus[min_ind] - lambdas[min_ind])

    kldiv = np.sum(kl_divergence(p_opt, p0))
    temp_R = ( (kldiv + np.log(E_s_Z/delta)) / N)
    Rg = np.sqrt(temp_R/2)
    L = expected_cost(p_opt, costs_precomputed)
    pac_bound = kl_inverse_l(L , temp_R, params) #KL inverse

    if params['print_stats']:
        print("  Cp =", np.round(Cp,2), ", Rg =", np.round(Rg,2))
        print("  minimize Cp + Rg ->", np.round(tau_opt,5))
        print("  KL inverse for tighter bound ->", np.round(pac_bound,5))

    return p_opt, pac_bound

#Opt extended robustness pac bound
def optimize_PAC_bound_MPB(costs_precomputed, p0, delta, N, params):
    B = calculate_B(params)
    # Number of actions
    L = len(p0)

    # Discretize lambdas - Different from original pac bound
    lambdas = np.linspace(0,np.e-1,100)

    # Initialize vectors for storing optimal solutions
    taus = np.zeros(len(lambdas))
    ps = len(lambdas)*[None]
    E_s_Z = 2*np.sqrt(N)

    for k in range(len(lambdas)):

        lambda0 = lambdas[k]

        # Create cost function variable
        tau = cvx.Variable()

        # Create variable for probability vector
        p = cvx.Variable(L)
        #Different from original pac bound
        cost_empirical = (1/N)*cvx.sum(np.exp(costs_precomputed)*p)

        # Constraints - Different from original pac bound
        #Making sure to enforce that p in [0,1]
        constraints = [lambda0**2 >= ( (cvx.sum(cvx.kl_div(p, p0)) + np.log(E_s_Z/delta)) / (2*N) )*(np.e-1)**2, lambda0 == (tau - cost_empirical), p >= 0, cvx.sum(p) == 1]

        prob = cvx.Problem(cvx.Minimize(tau), constraints)

        # Solve problem
        opt = prob.solve(verbose=False, solver=params['solver'])

        # Store optimal value and optimizer - Different from original pac bound
        if (opt != None):
            if (opt > np.e):
                taus[k] = np.e
            else:
                taus[k] = opt
            ps[k] = p.value

    # Find minimizer - Different from oroginal pac bound
    min_ind = np.argmin(taus)
    taus_transformed = np.log(taus) + B
    tau_opt_transformed = taus_transformed[min_ind]
    p_opt = ps[min_ind] #not taken out of exp form #need to transform p_opt first?
    Cpe = taus[min_ind] - lambdas[min_ind] #in exp form
    Cp = (Cpe - 1)/(np.e-1)

    kldiv = np.sum(kl_divergence(p_opt, p0))
    temp_R = ( (kldiv + np.log(E_s_Z/delta)) / N)
    Rg = np.sqrt(temp_R/2)


    L = expected_cost(p_opt, costs_precomputed)
    Le = expected_cost(p_opt, np.exp(costs_precomputed))

    #REP to get tighter bound
    pac_bound = optimize_REP_triangle_inequality(L, temp_R, B, params)

    if params['print_stats']:
        print("  B =", np.round(B,5), ", Cpe =", np.round(Cpe,2),", Cp =", np.round(Cp,2), ", Rg =", np.round(Rg,2))
        print("  minimize B + ln(Cpe + Rg(e-1)) ->", np.round(tau_opt_transformed,5))
        print("  REP for tighter bound ->", np.round(pac_bound,5))

    return p_opt, pac_bound
