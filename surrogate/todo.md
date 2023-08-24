
# shunqing

1. **Prediction**
    - [x] Conv-NN comparison
    - [x] Change depthN to head: 
        - [x] norm_hmin bug fixed
        - [ ] cannot predict depth well with large norm range
            - renorm and clip head in loss calculation
            - set weights in mse calculation: weird jumps in GAT

# astlingen

1. **Prediction**
    - [x] Conv-NN comparison
    - [x] if_flood comparison
    - [x] 60 step sequence: 
        - [x] Not well probably the normhmin bug
        - [x] Much better but water head bug same as shunqing: Use depthN instead

2. **MPC**
    - [x] Setting duration
    - [ ] 101 tests of 30s
    - [ ] Test 15/30 control interval with 5 setting duration
    - [ ] Genetic algorithm
        - [x] Discrete
        - [ ] Continuous
    - [ ] inversion of NN in continuous action space
        - training with continuous controller
        - LHS sampling
        - Gradient descent of each control action
        - termination: n_gen / convergence
        - Pick the global optimal action
    - [x] auto-differentiation through get_flood
        - [ ] need to rewrite get_flood with tensorflow
        - [ ] need to calculate objective function inside tf.GradientTape
    <!-- - [ ] Mating could not produce the required number of (unique) offsprings -->

# hague

1. **Prediction**
    - [x] Tide input
    - [x] Accurate cuminflow prediction
        - [x] Change data source to exclude unsteady pre-steps
    - [ ] Try 1-min interval
    - [x] Check: No flooding in all rainfalls: only flood in several rainfalls (sort intensity)

2. **MPC**
    - [ ] Continuous action space
        - [ ] GA
        - [ ] Gradient-based search
