# PAC-Bayes-Control - code for extension: Controllers with Robustness to Domain Shift

# Requirements

PyBullet: https://pybullet.org/wordpress/

CVXPY: https://github.com/cvxgrp/cvxpy

Solver: https://www.mosek.com/downloads/9.0.79/

# Running the code

The script main.py will run everything. In particular, it will:

- Optimize the PAC-Bayes controller using Relative Entropy Programming (REP).

- Optimize the extended robustness PAC-Bayes controller using REP.

- Estimate the true expected cost for the optimized controller on novel environments in order to compare with the PAC-Bayes bound (In order to speed things up, we've included a file with precomputed costs that you can load from.)

- Visualize the controller running on a number of test environments.

Description of the other files:

optimize_PAC_bound.py: Solves the Relative Entropy Program for optimizing PAC-Bayes controllers.

utils.py: Contains a number of utility functions for performing simulations (e.g., generating obstacle environments, implementing the robot's dynamics, simulating the depth sensor, etc.) as well as computation utility functions (e.g. calculating KL divergence, expected cost, etc.)

# Saving/Loading precomputed costs

Folder nobs7 refers to environments with 7 obstacles, subfolder r10-110 refers to environments with also a radius between 0.1 and 1.1 meters. With these two settings and folders, a file called costsprecomp_... will be generated that includes the number of environments used to compute the controllers, the seed, and 1000 times the KL divergence between train and test distributions.

- As long as there is a folder for the number of obstacles and a subfolder for the radius range, files will be loaded and saved automatically.

- Specifications for saving/loading files are in precompute_all_costs in main.py
