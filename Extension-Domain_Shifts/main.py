import pickle
from optimize_PAC_bound import *

#Main Functions

#Setup pybullet interface
def setup_pybullet(GUI, params):
    robotRadius = params['robotRadius']

    if (GUI):
        pybullet.connect(pybullet.GUI)
        visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=robotRadius,rgbaColor=[0,0,0,0]) # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
    else:
        pybullet.connect(pybullet.DIRECT)
        visualShapeId = -1

    pybullet.loadURDF("../URDFs/plane.urdf") # Ground plane
    husky = pybullet.loadURDF("../URDFs/husky.urdf", globalScaling=0.5) # Load robot from URDF
    colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=robotRadius) # Sphere
    mass = 0
    sphere = pybullet.createMultiBody(mass,colSphereId,visualShapeId)

    return husky, sphere

#Load/compute costs on environments
def precompute_all_costs(numEnvs, rseeds, K, L, husky, sphere, GUI, dataset, params):
    if params['case'] == 'radius_beta_distrib':
        folder = "nobs" + str(params['numObs']) + "/r" + str(int(params['XNPrange'][0]*100)) + "-" + str(int(params['XNPrange'][1]*100))
    else: folder=""

    if params['load_data'] == 0:
        costs_precomputed = {}
        print("\n\n")
        for i in range(len(process)):
            print("Precomputing costs for", process[i], "on", dataset[i])
            costs_precomputed[i] = precompute_environment_costs(numEnvs[i], K, L, params, husky, sphere, GUI[i], rseeds[i], dataset[i])

        f = open(folder+"/costsprecomp_nenvs" + str(params['p_nEnvs']) + "_seed" + str(params['seed']) + "_beta" + str(int(calculate_B(params)*1000)) + ".pickle","wb")
        pickle.dump(costs_precomputed, f)
        f.close()
    else: #params['load_data'] == 1
        print("\n")
        for i in range(len(process)):
            print("Loading costs for", process[i], "on", dataset[i])
        f = open(folder+"/costsprecomp_nenvs" + str(params['p_nEnvs']) + "_seed" + str(params['seed']) + "_beta" + str(int(calculate_B(params)*1000)) + ".pickle","rb")
        costs_precomputed = pickle.load(f)

    return costs_precomputed

#loading in robot parameters
def get_parameters(params):
    params['numRays'] = 20 # number of rays for sensor measurements
    params['senseRadius'] = 5.0 # sensing radius
    params['robotRadius'] = 0.27 # radius of robot
    params['robotHeight'] = 0.15/2 # rough height of COM of robot
    params['th_min'] = -np.pi/3 # sensing angle minimum
    params['th_max'] = np.pi/3 # sensing angle maximum
    params['T_horizon'] = 100 # time horizon over which to evaluate everything
    params['thetas_nominal'] = np.reshape(np.linspace(params['th_min'], params['th_max'], params['numRays']), (params['numRays'],1))

    params['solver'] = cvx.MOSEK
    params['verbose'] = False

    params['x_lim'] = [-5.0, 5.0] # Distribution parameters
    params['y_lim'] = [0.0, 10.0]

    if params['case'] == 'fixed_radius_avg_beta': #can be used for plotting
        [a0,b0] = [0.8, 1.25]
        [a1,b1] = [1.0, 1.0]
        [x0,x1] = [0.1, 0.35]
        params['avgrtrain'] = avg0 = a0 / (a0 + b0) * (x1-x0) + x0
        params['avgrtest'] = avg1 = a1 / (a1 + b1) * (x1-x0) + x0
        print("Switching to average radius")
    elif params['case'] == 'radius_beta_distrib':
        #params['XNPtrain'] = [2,2] #[alpha, beta]
        #params['XNPtest'] = [2,2] #B = 0
        params['XNPtrain'] = [0.8, 1.25]
        params['XNPtest'] =  [1.0, 1.0] #B=0.081(8)
        params['numObs'] = 7
        params['XNPrange'] = [0.1, 1.1]
    else:
        print("Not able to handle this case")
        exit(1)


################################################################################
params = {} # Initialize parameter dictionary
params['seed'] = 10
params['case'] = 'radius_beta_distrib' # Radius of obstacles drawn from a beta distribution
params['print_stats'] = 1 # Do you want to print information about the optimization?
params['p_nEnvs'] = 1000 # Number of environments
params['load_data'] = 1 # Are you loading data from a file?

get_parameters(params) # Loading in some robot parameters
np.random.seed(seed=params['seed'])

#pseudo code
# (1, 3, 5) estimate true cost with 10^5 environments
#   - (train, test - drawn from different distributions)
# (0) run normal pac bound on train - things are drawn from the same distribution
# (2) run normal pac bound on test - as if things are drawn from the same distribution
# (4) run modified pac bound on train to get bound for test

GUI = [False, False, False, False, False, False] #which parts of the process do you want to see
numEnvs = [params['p_nEnvs'], 10000, params['p_nEnvs'], 10000, params['p_nEnvs'], 10000] #Number of environments in each case
rseeds = np.int32(10*5*np.random.random_sample(9)) #seeds
dataset = ['train', 'train', 'test', 'test', 'train', 'test']
process = ['PAC-Bayes bound', 'estimating true cost', 'PAC-Bayes bound', 'estimating true cost', 'Extended robustness PAC-Bayes bound', 'estimating true cost']
delta = 0.01 # PAC failure probability
K, L = create_L_controllers(5, 10, params['numRays'], params['thetas_nominal']) #Creatining L controllers
p0 = L*[1.0/L] #prior for PAC-Bayes
husky, sphere = setup_pybullet(GUI[0], params) # pyBullet

#display car on both train and test environments using controllers we made
GUI_learnedcontrollers = True #Do you want to see the controllers
numEnvs_learnedcontrollers = 10 #How many examples of each

################################################################################
#Estimate true cost for all the environments
costs_precomputed = precompute_all_costs(numEnvs, rseeds, K, L, husky, sphere, GUI, dataset, params)

#Run pac bound optimizer on train
print("\nOptimizing controller for TRAIN using PAC-Bayes bound...")
p_opt_pac_train_NPB, pac_bound_train_NPB = optimize_PAC_bound_NPB(costs_precomputed[0], p0, delta, numEnvs[0], params)
print("\nEstimating true cost for train with PAC-Bayes bound controller")
true_cost_pac_train_NPB = expected_cost(p_opt_pac_train_NPB, costs_precomputed[1])

#Run pac bound optimizer on test
print("\nOptimizing controller for TEST using PAC-Bayes bound...")
p_opt_pac_test_NPB, pac_bound_test_NPB = optimize_PAC_bound_NPB(costs_precomputed[2], p0, delta, numEnvs[2], params)
print("\nEstimating true cost for test with PAC-Bayes bound controller")
true_cost_pac_test_NPB = expected_cost(p_opt_pac_test_NPB, costs_precomputed[3])

#Run extended robustness pac bound optimizer on train to get a bound for test
print("\nOptimizing controller for TEST using extended robustness PAC-Bayes bound...")
p_opt_pac_test_MPB, pac_bound_test_MPB = optimize_PAC_bound_MPB(costs_precomputed[4], p0, delta, numEnvs[4], params)
print("\nEstimating true cost for test with extended robustness PAC-Bayes bound controller")
true_cost_pac_test_MPB = expected_cost(p_opt_pac_test_MPB, costs_precomputed[5])
print("\nDone!")

#Print results
print("\n")
print("Cost bound on train with PAC-Bayes bound controller:                            ", np.round(pac_bound_train_NPB,5))
print("True cost estimate on train with PAC-Bayes bound controller (est with 10^5 env):", np.round(true_cost_pac_train_NPB,5))
print("Cost bound on test with PAC-Bayes bound controller:                             ", np.round(pac_bound_test_NPB,5))
print("True cost estimate on test with PAC-Bayes bound controller (est with 10^5 env): ", np.round(true_cost_pac_test_NPB,5))
print("")
print("Cost bound on test with extended robustness PAC-Bayes bound controller:                              ", np.round(pac_bound_test_MPB,5))
print("True cost estimate on test with extended robustness PAC-Bayes bound controller  (est with 10^5 env): ", np.round(true_cost_pac_test_MPB,5))
print("\n")

pybullet.disconnect()

################################################################################
if (GUI_learnedcontrollers): #Run optimized controller on some environments to visualize
    print("Setting up to display example environments with learned controllers")
    # Clean up probabilities
    p_opt_pac_train_NPB = np.maximum(p_opt_pac_train_NPB, 0)
    p_opt_pac_train_NPB = p_opt_pac_train_NPB/np.sum(p_opt_pac_train_NPB)

    p_opt_pac_test_NPB = np.maximum(p_opt_pac_test_NPB, 0)
    p_opt_pac_test_NPB = p_opt_pac_test_NPB/np.sum(p_opt_pac_test_NPB)

    p_opt_pac_test_MPB = np.maximum(p_opt_pac_test_MPB, 0)
    p_opt_pac_test_MPB = p_opt_pac_test_MPB/np.sum(p_opt_pac_test_MPB)

    setup_pybullet(True, params)

    print("Simulating (PAC-Bayes bound)-optimized controller on TRAIN ENVIRONMENTS")
    simulate_controller(numEnvs_learnedcontrollers, K, p_opt_pac_train_NPB, params, husky, sphere, True, rseeds[6], 'train')

    print("Simulating (PAC-Bayes bound)-optimized controller on TEST ENVIRONMENTS")
    simulate_controller(numEnvs_learnedcontrollers, K, p_opt_pac_test_NPB, params, husky, sphere, True, rseeds[7], 'test')

    print("Simulating (extended robustness PAC-Bayes bound)-optimized controller on TEST ENVIRONMENTS")
    simulate_controller(numEnvs_learnedcontrollers, K, p_opt_pac_test_MPB, params, husky, sphere, True, rseeds[8], 'test')

    pybullet.disconnect()

################################################################################
