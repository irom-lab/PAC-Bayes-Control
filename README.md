# PAC-Bayes-Control

Code for paper:

"PAC-Bayes Control: Synthesizing Controllers that Provably Generalize to Novel Environments"

Authors: Anirudha Majumdar and Maxwell Goldstein

The code here provides a complete implementation of the Relative Entropy Programming version of the approach presented in the paper on the obstacle avoidance example. 

# Requirements

Pybullet: https://pybullet.org/wordpress/

CVXPY: https://github.com/cvxgrp/cvxpy

SCS solver: https://github.com/cvxgrp/scs

# Running the code

The script main.py will run everything. In particular, it will:

- Optimize the PAC-Bayes controller using Relative Entropy Programming.
    
- Estimate the true expected cost for the optimized controller on novel environments in order to compare with the PAC-Bayes bound (In order to speed things up, we use a small number of environments here. A large number of samples should be used for a more accurate estimate, as is done in the paper.)
    
- Visualize the controller running on a number of test environments.
    
Description of the other files:

optimize_PAC_bound.py: Solves the Relative Entropy Program for optimizing PAC-Bayes controllers.

utils_simulation.py: Contains a number of utility functions for performing simulations (e.g., generating obstacle environments, implementing the robot's dynamics, simulating the depth sensor, etc.).

compute_kl_inverse.py: Self-contained implementation for computing the KL inverse using Relative Entropy Programming.