# GNN-UDS
A hydraulic surrogate model and real-time control methods of urban drainage networks.
 
 Please feel free to read or cite our paper below.

 **GNN-based model**: Zhang, Z., Tian, W., Lu, C., Liao, Z. and Yuan, Z. 2024. Graph neural network-based surrogate modelling for real-time hydraulic prediction of urban drainage networks. Water Research, 263, 122142. https://doi.org/10.1016/j.watres.2024.122142

 **Gradient-based MPC**: Zhang, Z., Tian, W., Liao, Z. and Yuan, Z. 2026. Differentiable neural network-based models enable gradient-based optimization for model predictive control of urban drainage networks. Water Research, 291, 125188. https://authors.elsevier.com/a/1mKOH9pi-huFn

## How-to
1. generate labels

    ```
    python main.py --simulate --env (env_name) --data_dir (data_name) (--act)
    ```

    Simulations are made to generate training data at `./envs/data/env_name/data_name/`.

2. training

    ```
    python main.py --train --env (env_name) --data_dir (data_name) --model_dir (model_name) (--act conti/rand) (--conv GAT) (--recurrent Conv1D) (--batch_size 64) (--epochs 50000) (--if_flood) (--seq 60) (--dim 128) (--nly 5)
    ```

    The model structure is built and trained with data at `data_dir` for epochs. Details of the model and training parameters refer to `config.yaml`. The trained model and training loss logging are saved at `./model/(env_name)/(model_name)/`.

3. testing

    ```
    python main.py --test --env (env_name) --model_dir (model_name) --rain_dir/rain_num/rain_suffix (rain_name, depends on the source) --result_dir (result_name) (--act conti/rand)
    ```

    The model is loaded to emulate the drainage network in various rainfalls. Details of the model and testing parameters refer to `./utils/config.yaml` and `parser` func at `main.py`. The testing states, performance (perfs), settings and prediction results of each rainfall are saved at `./results/(env_name)/(result_name)/`.

4. model predictive control

    ```
    python mpc.py --env (env_name) --act (conti/rand) (--ctrl_step 5) (--horizon 60) (--lag) --rain_dir/rain_num/rain_suffix (rain_name) --result_dir (result_name) (--use_current) (--pop_size 64) (--termination ftol 0.001) (--surrogate) (--model_dir model_name) (--gradient) (--method l-bfgs-b)
    ```

    Implement Model Predictive Control with physics-based/surrogate internal model and genetic algorithm/gradient-based method. Details of the MPC parameters refer to `./utils/mpc.yaml` and `parser` func at `mpc.py`. The states, performance (perfs), settings，and optimization results of each rainfall will be saved at `./results/(env_name)/(result_name)/`.


## Drainage networks
1. **shunqing**
   - Stormwater network
   - 113 nodes (105 junctions and 8 outfalls)
   - 131 conduits and 106 subcatchments (cover 33.02 km2)
   - 148 synthetic rainfalls included with duration of 6-24 hrs
   - Details refer to [ga_ann_for_uds](https://github.com/lhmygis/ga_ann_for_uds).
     
2. **astlingen**
   - Combined sewer network
   - 30 nodes (23 junctions, 6 tanks and 1 outfall)
   - 29 edges (23 conduits and 6 outflow orifices)
   - 10-yr rainfall monitoring data of 4 gauges are included
   - Details refer to [SWMM-Astlingen](https://github.com/open-toolbox/SWMM-Astlingen).

3. **chaohu**
   - Combined sewer network
   - 2 pump stations with storage tanks (CC and JK)
   - CC has 2 storm pumps and 2 sewage pumps
   - JK has 2 storm pumps and 1 sewage pump
   - Chicago rainfall pattern.

4. **hague**
   - Stormwater network
   - 2 detention ponds with outflow valves
   - 10-yr rainfall monitoring data are included
   - Details refer to [swmm_wq_rl](https://github.com/UVAdMIST/swmm_wq_rl).

## Requirements
- torch == 2.4.0
- torch_geometric == 2.6.1
- einops == 0.8.1
- pyswmm == 1.5.1
- pystorms == 1.0.0
- swmm-api == 0.4.66
- scipy == 1.15.0
- pymoo == 0.6.0
- matplotlib
- tensorboard
- reinmax