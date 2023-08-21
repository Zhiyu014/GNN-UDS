
# shunqing

1. **Prediction**
    - [x] Conv-NN comparison
    - [x] Change depthN to head: 
        - norm_hmin bug fixed
        - cannot predict depth well with large norm range

# astlingen

1. **Prediction**
    - [x] Conv-NN comparison
    - [x] if_flood comparison
    - [ ] 60 step sequence: Not well probably the normhmin bug
    - [ ] 101 tests of 30s

2. **MPC**
    - [x] Setting duration
    - [ ] inversion of NN in continuous action space
    - [ ] auto-differentiation through get_flood
    <!-- - [ ] Mating could not produce the required number of (unique) offsprings -->

# hague

1. **Prediction**
    - [x] Tide input
    - [ ] Accurate cuminflow prediction: Change data source from continuous simulations
    - [ ] Check: No flooding in all rainfalls: only flood in several rainfalls (sort intensity)

2. **MPC**
    - [ ] Continuous action space
