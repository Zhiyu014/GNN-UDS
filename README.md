# GNN-UDS
 A GNN-based surrogate model of urban drainage networks.

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
1. **astlingen**
   - Combined sewer network
   - 30 nodes (23 junctions, 6 tanks and 1 outfall)
   - 29 edges (23 conduits and 6 outflow orifices)
   - 10-yr rainfall monitoring data of 4 gauges are included
   - Details refer to [SWMM-Astlingen](https://github.com/open-toolbox/SWMM-Astlingen).

3. **shunqing**
   - Stormwater network
   - 113 nodes (105 junctions and 8 outfalls)
   - 131 conduits and 106 subcatchments (cover 33.02 km2)
   - 148 synthetic rainfalls included with duration of 6-24 hrs
   - Details refer to [ga_ann_for_uds](https://github.com/lhmygis/ga_ann_for_uds).

5. **RedChicoSur**
   - Stormwater network
   - 443 nodes (442 junctions and 1 outfall)
   - 444 edges (390 conduits and 54 orifices)
   - 2-hr synthetic rainfalls (Chicago hytograph) included
   - Details refer to [MatSWMM](https://github.com/gandresr/MatSWMM).

## Requirements
- tensorflow == 2.6.0
- spektral == 1.2.0
- pystorms == 1.0.0
- swmm-api == 0.2.0.18.3
- matplotlib == 3.5.2
- pymoo == 0.6.0
