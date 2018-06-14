# Main script for "PAC-Bayes Control: Synthesizing Controllers that Provably Generalize to Novel Environments"
# Implements Relative Entropy Programming version of approach
# Requirements:
# - PyBullet: https://pybullet.org/wordpress/
# - SCS: https://github.com/cvxgrp/scs

import pybullet as pybullet
import numpy as np
import time

from optimize_PAC_bound import *
from utils_simulation import *

###########################################################################################
# Functions

# Utility function for computing KL divergence
def kl_divergence(p,q):
    L = len(p)
    
    kl_p_q = 0.0
    for l in range(0,L):
        
        if (p[l] > 1e-6): # if p[l] is close to 0, then 0*log(0) := 0
            kl_p_q = kl_p_q + p[l]*np.log((p[l])/(q[l]))
        
    return kl_p_q    

# Compute cost function
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

# Precompute costs for different environments and controllers
def precompute_environment_costs(numEnvs, K, L, params, husky, sphere, GUI, seed):
    
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
    
    for env in range(0,numEnvs):
                
        # Print
        if (env%10 == 0):
            print(env, "out of", numEnvs)
    
        # Sample environment
        heightObs = 20*robotHeight
        obsUid = generate_obstacles(pybullet, heightObs, robotRadius)  

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

                if (GUI):
                    pybullet.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[state[0], state[1], 2*robotHeight])

                    time.sleep(0.025) 


                # Check if the robot is in collision. If so, cost = 1.0.      
                # Get closest points. Note: Last argument is distance threshold. Since it's set to 0, the function will only return points if the distance is less than zero. So, closestPoints is non-empty iff there is a collision.
                closestPoints = pybullet.getClosestPoints(sphere, obsUid, 0.0)


                # See if the robot is in collision. If so, cost = 1.0. 
                if closestPoints: # Check if closestPoints is non-empty 
                    cost_env_l = 1.0
                    break # break out of simulation for this environment

            
            # Check that cost is between 0 and 1 (for sanity)
            if (cost_env_l > 1.0):
                raise ValueError("Cost is greater than 1!")
                
            if (cost_env_l < 0.0):
                raise ValueError("Cost is less than 0!")
            
            # Record cost for this environment and this controller
            costs[env][l] = cost_env_l
            
        # Remove obstacles
        pybullet.removeBody(obsUid)
    
    return costs
    
# Compute expected cost (just a utility function)
def expected_cost(p, precomputed_costs):
    
    # Compute average cost
    costs_envs = np.matmul(precomputed_costs, p)
    cost_mean = np.mean(costs_envs)
    
    return cost_mean

# Compute control input    
def compute_control(y, K):
    # y is vector of depth measurements
    
    u_diff = np.matmul(1./y, K)
       
    return u_diff  

def simulate_controller(numEnvs, K, p_opt, params, husky, sphere, GUI, seed):
    
    # Parameters
    numRays = params['numRays']
    senseRadius = params['senseRadius']   
    robotRadius = params['robotRadius']
    robotHeight = params['robotHeight']
    thetas_nominal = params['thetas_nominal']
    T_horizon = params['T_horizon']
    
    # Fix random seed for consistency of results
    np.random.seed(seed)
    
    for env in range(0,numEnvs):
                
        # Print
        if (env%100 == 0):
            print(env, "out of", numEnvs)
    
        # Sample environment
        heightObs = 20*robotHeight
        obsUid = generate_obstacles(pybullet, heightObs, robotRadius)  
        
        # Choose controller by sampling
        l = np.random.choice(range(0,len(p_opt_pac)), 1, p=p_opt_pac)
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

            if (GUI):
                pybullet.resetDebugVisualizerCamera(cameraDistance=5.0, cameraYaw=0.0, cameraPitch=-45.0, cameraTargetPosition=[state[0], state[1], 2*robotHeight])

                time.sleep(0.025) 


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

    
###########################################################################################

# Initial setup

# Flag that sets if things are visualized
# GUI = True; # Only for debugging purposes
GUI = False;

# PAC parameters
m = 100 # Number of environments to sample
delta = 0.01 # PAC failure probability

# Random seed (for consistency of results)
random_seed = 5 

# pyBullet
if (GUI):
    pybullet.connect(pybullet.GUI)
else:
    pybullet.connect(pybullet.DIRECT) 

# Get some robot parameters
params = get_parameters()
robotRadius = params['robotRadius']
numRays = params['numRays']
thetas_nominal = params['thetas_nominal']

# Ground plane
pybullet.loadURDF("./URDFs/plane.urdf")

# Load robot from URDF
husky = pybullet.loadURDF("./URDFs/husky.urdf", globalScaling=0.5)

# Sphere
colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=robotRadius)
mass = 0
if (GUI):
    visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=robotRadius,rgbaColor=[0,0,0,0]) # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)
else:
    visualShapeId = -1

sphere = pybullet.createMultiBody(mass,colSphereId,visualShapeId)

################################################################################
# Controller and optimization setup

# Choose L controllers
num_x_intercepts = 5
num_y_intercepts = 10
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
                

################################################################################
# Prior for PAC-Bayes

p0 = L*[1.0/L] # Probability vector corresponding to prior (should sum to 1).                
                

################################################################################
# Perform optimizations
# Do precomputations for optimization
print(" ")
print(" ")
print("Precomputing costs for environments (could/should easily be parallelized)...")
tic = time.time()
# Precompute costs for m different environments and L controllers
costs_precomputed = precompute_environment_costs(m, K, L, params, husky, sphere, GUI, random_seed)
toc = time.time()
time_precomputation = toc-tic
print("Done precomputing.")

# Optimize with PAC-Bayes bound
tic = time.time()
print("Optimizing controller (PAC-Bayes)...")
# Using Relative Entropy Programming
bound_REP, p_opt_pac, taus = optimize_PAC_bound(costs_precomputed, p0, delta)

# Save results
# np.savez('results_REP', p_opt_pac=p_opt_pac, K=K)

# Compute KL inverse bound
pac_bound = kl_inverse(expected_cost(p_opt_pac, costs_precomputed), (kl_divergence(p_opt_pac,p0)+np.log(2*np.sqrt(m)/delta))/m) 
print("Done optimizing.")
toc = time.time()
time_optimization = toc-tic
################################################################################

################################################################################
# Estimate true expected cost
print("Precomputing costs for estimating true cost...")
# Precompute costs for different environments and controllers
numEnvs = 100 # 100000 were used in the paper (here, we're using few environments to speed things up) 
seed = 10 # Different random seed
costs_precomputed_for_est = precompute_environment_costs(numEnvs, K, L, params, husky, sphere, GUI, seed)
print("Done precomputing.")

print("Estimating true cost for PAC-Bayes controller based on ", numEnvs, " environments...")
true_cost_pac = expected_cost(p_opt_pac, costs_precomputed_for_est)
print("Done estimating true cost.")
################################################################################

# Print results
print("Time for precomputation: ", time_precomputation)
print("Time for optimization: ", time_optimization)
print("Cost bound (PAC-Bayes): ", pac_bound)
print("True cost estimate (PAC-Bayes): ", true_cost_pac)

# Disconect from pybullet
pybullet.disconnect()


################################################################################
# Run optimized controller on some environments to visualize

# Flag that sets if things are visualized
GUI = True; 
pybullet.connect(pybullet.GUI)

# Clean up probabilities
p_opt_pac = np.maximum(p_opt_pac, 0)
p_opt_pac = p_opt_pac/np.sum(p_opt_pac)

random_seed = 25 #
numEnvs = 10 # Number of environments to show videos for

# Ground plane
pybullet.loadURDF("./URDFs/plane.urdf")

# Load robot from URDF
husky = pybullet.loadURDF("./URDFs/husky.urdf", globalScaling=0.5)

# Sphere
colSphereId = pybullet.createCollisionShape(pybullet.GEOM_SPHERE, radius=robotRadius)
mass = 0

visualShapeId = pybullet.createVisualShape(pybullet.GEOM_SPHERE, radius=robotRadius,rgbaColor=[0,0,0,0]) # This just makes sure that the sphere is not visible (we only use the sphere for collision checking)


sphere = pybullet.createMultiBody(mass,colSphereId,visualShapeId)

print("Simulating optimized controller in a few environments...")

# Play videos
simulate_controller(numEnvs, K, p_opt_pac, params, husky, sphere, GUI, random_seed)

# Disconect from pybullet
pybullet.disconnect()

print("Done.")












   