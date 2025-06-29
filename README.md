# GNN-UDS
 A hydraulic surrogate model and real-time control methods of urban drainage networks. 
 
 Please feel free to read or cite our paper below.

**GNN-based model**: Zhang, Z., Tian, W., Lu, C., Liao, Z. and Yuan, Z. 2024. Graph neural network-based surrogate modelling for real-time hydraulic prediction of urban drainage networks. Water Research, 263, 122142. https://doi.org/10.1016/j.watres.2024.122142

## How-to
1. generate labels

    ```
    python main.py --simulate --env (env_name) --data_dir (data_name) (--edge_fusion) (--act)
    ```

    Simulations are made to generate training data at `./envs/data/env_name/data_name/`.

2. training

    ```
    python main.py --train --env (env_name) --data_dir (data_name)  --model_dir (model_name) (--edge_fusion) (--act) (--conv GAT) (--recurrent Conv1D) (--batch_size 64) (--epochs 20000) (--if_flood) (--norm) (--resnet) (--seq_in 10) (--seq_out 10)
    ```

    The model structure is built and trained with data at `data_dir` for epochs. Details of the model and training parameters refer to `config.yaml`. The trained model and training loss logging are saved at `./model/env_name/model_name/`.

3. testing

    ```
    python main.py --test --env (env_name) --model_dir (model_name) --result_dir (result_name) (--edge_fusion) (--act) (--conv GAT) (--recurrent Conv1D) (--if_flood) (--norm) (--resnet) (--seq_in 10) (--seq_out 10)
    ```

    The model is loaded to emulate the drainage network in various rainfalls. Details of the model and testing parameters refer to `config.yaml` and `parser` func at `main.py`. The testing states, performance (perfs), settings and prediction results of each rainfall are saved at `./result/env_name/result_name/`.


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


## Requirements
- tensorflow == 2.6.0
- keras == 2.6.0
- tensorflow_probability == 0.11.0
- spektral == 1.2.0
- protobuf == 3.20.0
- pyswmm == 1.5.1
- pystorms == 1.0.0
- swmm-api == 0.2.0.18.3
- matplotlib == 3.5.2
- pymoo == 0.6.0
