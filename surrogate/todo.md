
# shunqing

1. **Prediction**
    - [x] Conv-NN comparison
    - [x] Change depthN to head: 
        - [x] norm_hmin bug fixed
        - [x] cannot predict depth well with large norm range
            - renorm and clip head in loss calculation
            - set weights in mse calculation: weird jumps in GAT
    - [x] if-flood comparison: no change
    - [x] edge fusion comparison: no change
    - [x] loss using flooding volume (bal): if it increases flooding prediction in astlingen
        - it truely rmse and mae but classification metrics are not good
        - large error water head
        - [x] try on water depth in astlingen: not good

# astlingen

1. **Prediction**
    - [x] Conv-NN comparison
    - [x] if_flood comparison: it seems that no-flood > flood > bal
    - [x] 60 step sequence: 
        - [x] Not well probably the normhmin bug
        - [x] Much better but water head bug same as shunqing: Use depthN instead

2. **MPC**
    - [x] Setting duration
    - [ ] 101 tests of 30s
    - [ ] Test no flood
    - [ ] Test 15/30 control interval with 5 setting duration
        - 15 ctrl_inte not work well with 60 prediction steps and 7 act
        - 5 ctrl_inte is better
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
