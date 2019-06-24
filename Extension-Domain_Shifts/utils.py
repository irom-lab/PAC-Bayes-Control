import pybullet as pybullet
import numpy as np
import cvxpy as cvx
import scipy, mosek

#Calculation Utilities
def expected_cost(p, precomputed_costs):

    # Compute average cost
    costs_envs = np.matmul(precomputed_costs, p)
    cost_mean = np.mean(costs_envs)

    return cost_mean

def kl_divergence(p,q):
    if hasattr(p, "__len__")!=True:
        p=[p, 1-p]; q=[q, 1-q] #Overloaded notation for Bernoulli distribution

    L = len(p)

    kl_p_q = 0.0
    for l in range(0,L):

        if (p[l] > 1e-6): # if p[l] is close to 0, then 0*log(0) := 0
            kl_p_q = kl_p_q + p[l]*np.log((p[l])/(q[l]))

    return kl_p_q

def kl_inverse_l(q, c, params): #left kl inverse
    # KL(q||p) <= c
    # KLinv(q||c) = p
    # solve: sup  p
    #       s.t.  KL(q||p) <= c
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q,1-q])

    constraints = [c >= cvx.sum(cvx.kl_div(q_bernoulli,p_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    opt = prob.solve(verbose=params['verbose'], solver=params['solver'])

    return p_bernoulli.value[0]

def kl_inverse_r(c, q, params): #right kl inverse
    # KL(q||p) <= c
    # KLinv(c||p) = q
    # solve: sup  q
    #       s.t.  KL(q||p) <= c
    p_bernoulli = cvx.Variable(2)

    q_bernoulli = np.array([q,1-q])

    constraints = [c >= cvx.sum(cvx.kl_div(p_bernoulli, q_bernoulli)), 0 <= p_bernoulli[0], p_bernoulli[0] <= 1, p_bernoulli[1] == 1.0-p_bernoulli[0]]

    prob = cvx.Problem(cvx.Maximize(p_bernoulli[0]), constraints)

    # Solve problem
    opt = prob.solve(verbose=params['verbose'], solver=params['solver'])

    return p_bernoulli.value[0]

def calculate_B(params): #Calculates the KL divergence between the two environment distributions
    B = 10**10 #something that will indicate there is a problem if not changed
    if params['case'] == 'gaussian-1D':
        m1, s1 = params['XNPtrain']
        m2, s2 = params['XNPtest']
        #beta upper bounded by KL divergence between two non-truncated gaussians
        #So we can just calculate the divergence between gaussians and use that as a valid upper bound
        B = np.log(s2/s1) + (s1**2 + (m1 - m2)**2)/(2*s2**2) - 1/2
        B = params['numObs']*B #we would like to avoid this numObs
    elif params['case'] == 'radius_beta_distrib':
        a0, b0 = params['XNPtest']
        a1, b1 = params['XNPtrain']
        B = np.log(scipy.special.beta(a1,b1)/scipy.special.beta(a0,b0))
        B += (a0-a1)*scipy.special.digamma(a0)
        B += (b0-b1)*scipy.special.digamma(b0)
        B += (a1-a0 + b1-b0)*scipy.special.digamma(a0 + b0)
        factor = 1 / (params['XNPrange'][1] - params['XNPrange'][0])
        B *= factor

    return B

def optimize_REP_triangle_inequality(L, Rg, B, params, tol=10**-8): #For extended robustness
    # max.  L_t over L_t, Ltr
    # s.t   KL(L||L_tr) <= Rg
    #       KL(L_t||L_tr) <= B
    #       0 <= L_t <= 1
    #       0 <= L_tr <= 1
    L_t = cvx.Variable()
    L_tr = cvx.Variable()
    cs1 = cvx.kl_div(L, L_tr) + cvx.kl_div(1-L, 1-L_tr) <= Rg
    cs2 = cvx.kl_div(L_t, L_tr) + cvx.kl_div(1-L_t, 1-L_tr) <= B
    constraints_pacbound = [cs1, cs2, L_t <= 1, L_t >= 0, L_tr <= 1, L_tr >= 0]
    prob = cvx.Problem(cvx.Maximize(L_t), constraints_pacbound)
    pac_bound = prob.solve(verbose=params['verbose'], solver=params['solver'])

    return pac_bound

#Simulation Utilities
def cost_fcn(p, p0, delta, precomputed_costs, pac_bayes):

    # m: number of different environments (i.e., # of samples)
    m = np.shape(precomputed_costs)[0]

    # Compute average cost
    costs_envs = np.matmul(precomputed_costs, p)
    cost = np.mean(costs_envs)

    if (pac_bayes): # If we're applying PAC-Bayes theory

        # Now compute KL term
        kl_p_p0 = kl_divergence(p,p0)

        # "Regularizer" term (this holds for all [0,1]-bounded loss functions [Maurer '04])
        cost_kl = np.sqrt((kl_p_p0 + np.log((2*np.sqrt(m))/delta))/(2*m))

        cost = cost + cost_kl

    return cost

def compute_control(y, K):
    return np.matmul(1./y, K)

def simulate_controller(numEnvs, K, p_opt, params, husky, sphere, GUI, seed, dataset):

    # Parameters
    numRays = params['numRays']
    senseRadius = params['senseRadius']
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    thetas_nominal = params['thetas_nominal']
    T_horizon = params['T_horizon']

    # Fix random seed for consistency of results
    np.random.seed(seed)

    if (GUI):
        pybullet.resetDebugVisualizerCamera(cameraDistance=10.0, cameraYaw=0.0, cameraPitch=-75.0, cameraTargetPosition=[0,5,0])

    for env in range(0,numEnvs):

        # Print
        if (env%100 == 0):
            print(env, "out of", numEnvs)

        # Sample environment
        heightObs = 20*robotHeight
        obsUid = generate_obstacles(pybullet, heightObs, params, dataset)

        # Choose controller by sampling
        l = np.random.choice(range(0,len(p_opt)), 1, p=p_opt)
        l = l[0]

        # Initialize position of robot
        state = [0.0, 1.0, 0.0] # [x, y, theta]
        quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

        pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
        pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])


        for t in range(0, T_horizon):

            # Get sensor measurement
            y = getDistances(pybullet, state, robotHeight, numRays, senseRadius, thetas_nominal)

            # Compute control input
            u = compute_control(y, K[l])

            # Update state
            state = robot_update_state(state, u)
            # print(state)

            # Update position of pybullet object
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
            pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
            pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])


            # Check if the robot is in collision. If so, cost = 1.0.
            # Get closest points. Note: Last argument is distance threshold. Since it's set to 0, the function will only return points if the distance is less than zero. So, closestPoints is non-empty iff there is a collision.
            closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)


            # See if the robot is in collision. If so, cost = 1.0.
            if closestPoints: # Check if closestPoints is non-empty
                # cost_env_l = 1.0
                break # break out of simulation for this environment


        # Remove obstacles
        pybullet.removeBody(obsUid)

    return True

def generate_obstacles(p, heightObs, params, dataset):
    robotRadius = params['robotRadius']
    x_lim = params['x_lim']
    y_lim = params['y_lim']
    numObs = params['numObs']

    linkMasses = [None]*(numObs+3) # +3 is because we have three bounding walls
    colIdxs = [None]*(numObs+3)
    visIdxs = [None]*(numObs+3)
    posObs = [None]*(numObs+3)
    orientObs = [None]*(numObs+3)
    parentIdxs = [None]*(numObs+3)
    linkInertialFramePositions = [None]*(numObs+3)
    linkInertialFrameOrientations = [None]*(numObs+3)
    linkJointTypes = [None]*(numObs+3)
    linkJointAxis = [None]*(numObs+3)

    for obs in range(numObs+3):
        linkMasses[obs] = 0.0
        visIdxs[obs] = -1
        parentIdxs[obs] = 0
        linkInertialFramePositions[obs] = [0,0,0]
        linkInertialFrameOrientations[obs] = [0,0,0,1]
        linkJointTypes[obs] = p.JOINT_FIXED
        linkJointAxis[obs] = np.array([0,0,1])
        orientObs[obs] = [0,0,0,1]

    #Adding obstacles to environment
    if params['case'] == 'radius_beta_distrib':
        scale = (params['XNPrange'][1] - params['XNPrange'][0])
        shift = params['XNPrange'][0]
        if dataset == 'train': #We use different distributions for the train/test environments
            radiusObs = shift + np.random.beta(params['XNPtrain'][0],params['XNPtrain'][1])*scale
        elif dataset == 'test':
            radiusObs = shift + np.random.beta(params['XNPtest'][0],params['XNPtest'][1])*scale
        for obs in range(numObs): #Adding the cylindrical obstacles
            posObs_obs = np.array([None]*3)
            posObs_obs[0] = x_lim[0] + (x_lim[1] - x_lim[0])*np.random.random_sample(1)
            posObs_obs[1] = 2.0 + y_lim[0] + (y_lim[1] - y_lim[0] - 2.0)*np.random.random_sample(1) # Push up a bit
            posObs_obs[2] = 0 # set z at ground level
            posObs[obs] = posObs_obs # .tolist()
            colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER,radius=radiusObs,height=heightObs)
    if params['case'] == 'fixed_radius_avg_beta': #if we want obstacles fixed at the average radius for the beta distribution
        radiusObs = params['avgr'+dataset]
        for obs in range(numObs): #Adding the cylindrical obstacles
            posObs_obs = np.array([None]*3)
            posObs_obs[0] = x_lim[0] + (x_lim[1] - x_lim[0])*np.random.random_sample(1)
            posObs_obs[1] = 2.0 + y_lim[0] + (y_lim[1] - y_lim[0] - 2.0)*np.random.random_sample(1) # Push up a bit
            posObs_obs[2] = 0 # set z at ground level
            posObs[obs] = posObs_obs # .tolist()
            colIdxs[obs] = p.createCollisionShape(p.GEOM_CYLINDER,radius=radiusObs,height=heightObs)

    # Left wall
    posObs[numObs] = [x_lim[0], (y_lim[0]+y_lim[1])/2.0, 0.0]
    colIdxs[numObs] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])

    # Right wall
    posObs[numObs+1] = [x_lim[1], (y_lim[0]+y_lim[1])/2.0, 0.0]
    colIdxs[numObs+1] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (y_lim[1] - y_lim[0])/2.0, heightObs/2])

    # Back wall
    orientObs[numObs+2] = [0,0,np.sqrt(2)/2,np.sqrt(2)/2]
    posObs[numObs+2] = [(x_lim[0]+x_lim[1])/2.0, y_lim[0], 0.0]
    colIdxs[numObs+2] = p.createCollisionShape(p.GEOM_BOX, halfExtents = [0.1, (x_lim[1] - x_lim[0])/2.0, heightObs/2])

    obsUid = p.createMultiBody(baseCollisionShapeIndex = -1, baseVisualShapeIndex = -1, basePosition = [0,0,0], baseOrientation = [0,0,0,1], baseInertialFramePosition = [0,0,0], baseInertialFrameOrientation = [0,0,0,1], linkMasses = linkMasses, linkCollisionShapeIndices = colIdxs, linkVisualShapeIndices = visIdxs, linkPositions = posObs, linkOrientations = orientObs, linkParentIndices = parentIdxs, linkInertialFramePositions = linkInertialFramePositions, linkInertialFrameOrientations = linkInertialFrameOrientations, linkJointTypes = linkJointTypes, linkJointAxis = linkJointAxis)

    return obsUid

def precompute_environment_costs(numEnvs, K, L, params, husky, sphere, GUI, seed, dataset):

    # Parameters
    numRays = params['numRays']
    senseRadius = params['senseRadius']
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    thetas_nominal = params['thetas_nominal']
    T_horizon = params['T_horizon']

    # Fix random seed for consistency of results
    np.random.seed(seed)

    # Initialize costs for the different environments and different controllers
    costs = np.zeros((numEnvs, L))

    if (GUI):
        pybullet.resetDebugVisualizerCamera(cameraDistance=10.0, cameraYaw=0.0, cameraPitch=-75.0, cameraTargetPosition=[0,5,0])

    for env in range(0,numEnvs):

        # Print
        if (env%10 == 0):
            print(env, "out of", numEnvs)

        # Sample environment
        heightObs = 20*robotHeight
        obsUid = generate_obstacles(pybullet, heightObs, params, dataset)

        for l in range(0,L):

            # Initialize position of robot
            state = [0.0, 1.0, 0.0] # [x, y, theta]
            quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2

            pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
            pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])

            # Cost for this particular controller (lth controller) in this environment
            cost_env_l = 0.0

            for t in range(0, T_horizon):

                # Get sensor measurement
                y = getDistances(pybullet, state, robotHeight, numRays, senseRadius, thetas_nominal)

                # Compute control input
                u = compute_control(y, K[l])

                # Update state
                state = robot_update_state(state, u)

                # Update position of pybullet object
                quat = pybullet.getQuaternionFromEuler([0.0, 0.0, state[2]+np.pi/2]) # pi/2 since Husky visualization is rotated by pi/2
                pybullet.resetBasePositionAndOrientation(husky, [state[0], state[1], 0.0], quat)
                pybullet.resetBasePositionAndOrientation(sphere, [state[0], state[1], robotHeight], [0,0,0,1])

                # Check if the robot is in collision. If so, cost = 1.0.
                # Get closest points.
                closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)

                # See if the robot is in collision. If so, cost = 1.0.
                if closestPoints: # Check if closestPoints is non-empty
                    cost_env_l = 1.0
                    break # break out of simulation for this environment

            # Record cost for this environment and this controller
            costs[env][l] = cost_env_l

        # Remove obstacles
        pybullet.removeBody(obsUid)

    return costs

#Robot Utilities
def robot_update_state(state, u_diff):

    # State: [x,y,theta]
    # x: horizontal position
    # y: vertical position
    # theta: angle from vertical (positive is anti-clockwise)

    # Dynamics:
    # xdot = -(r/2)*(ul + ur)*sin(theta)
    # ydot = (r/2)*(ul + ur)*cos(theta)
    # thetadot = (r/L)*(ur - ul)

    # Robot parameters
    r = 0.1; # Radius of robot wheel
    L = 0.5; # Length between wheels (i.e., width of base)

    dt = 0.05
    v0 = 2.5 # forward speed

    # Saturate udiff
    u_diff_max = 0.5*(v0/r)
    u_diff_min = -u_diff_max
    u_diff = np.maximum(u_diff_min, u_diff)
    u_diff = np.minimum(u_diff_max, u_diff)

    ul = v0/r - u_diff;
    ur = v0/r + u_diff;

    new_state = [0.0, 0.0, 0.0]
    new_state[0] = state[0] + dt*(-(r/2)*(ul + ur)*np.sin(state[2])) # x position
    new_state[1] = state[1] + dt*((r/2)*(ul + ur)*np.cos(state[2])) # y position
    new_state[2] = state[2] + dt*((r/L)*(ur - ul))

    return new_state

def getDistances(p, state, robotHeight, numRays, senseRadius, thetas_nominal):

    # Get distances
    # rays emanate from robot
    raysFrom = np.concatenate((state[0]*np.ones((numRays,1)), state[1]*np.ones((numRays,1)), robotHeight*np.ones((numRays,1))), 1)

    thetas = (-state[2]) + thetas_nominal # Note the minus sign: +ve direction for state[2] is anti-clockwise (right hand rule), but sensor rays go clockwise

    raysTo = np.concatenate((state[0]+senseRadius*np.sin(thetas), state[1]+senseRadius*np.cos(thetas), robotHeight*np.ones((numRays,1))), 1)

    coll = p.rayTestBatch(raysFrom, raysTo)

    dists = np.zeros((1,numRays))
    for i in range(numRays):
        dists[0][i] = senseRadius*coll[i][2]

    return dists

def create_L_controllers(num_x_intercepts, num_y_intercepts, numRays, thetas_nominal):
    L = num_x_intercepts*num_y_intercepts

    x_intercepts = np.linspace(0.1, 5.0, num_x_intercepts)
    y_intercepts = np.linspace(0.0, 10.0, num_y_intercepts)

    K = L*[None]
    for i in range(num_x_intercepts):
        for j in range(num_y_intercepts):
            K[i*num_y_intercepts + j] = np.zeros((numRays,1))
            for r in range(numRays):
                if (thetas_nominal[r] > 0):
                    K[i*num_y_intercepts + j][r] = y_intercepts[j]*(x_intercepts[i] - thetas_nominal[r])/x_intercepts[i]
                else:
                    K[i*num_y_intercepts + j][r] = y_intercepts[j]*(-x_intercepts[i] - thetas_nominal[r])/x_intercepts[i]
    return K, L
