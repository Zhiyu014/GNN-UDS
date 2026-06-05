from .base import basescenario
import os
from itertools import product
import numpy as np
import torch as th
import pandas as pd


HERE = os.path.dirname(__file__)


class central(basescenario):
    r"""Central Scenario

    External-inflow-driven RTC case with four controlled orifices.
    The objective follows the PTW peak/smoothing recommendation generated
    from the uncontrolled central simulations.
    """

    def __init__(self, config_file=None, swmm_file=None, global_state=True, initialize=True):
        config_file = os.path.join(HERE, "..", "config", "central.yaml") \
            if config_file is None else config_file
        super().__init__(config_file, swmm_file, global_state, initialize)

    def _stage_cost_np(self, q_ptw, q_prev, q_w):
        obj = self.config["objective"]
        weights = obj["weights"]
        q_bar = np.concatenate([q_prev,q_ptw],axis=-1).mean(axis=-1, keepdims=True)
        flat_cost = weights["flat"] * (q_ptw - q_bar) ** 2
        # beta = float(obj.get("peak_smooth", 10.0))
        # z = beta * (q_ptw - q_bar)
        # m = np.maximum(np.max(z, axis=-1, keepdims=True), 0.0)
        # smooth_peak = (m + np.log(np.exp(-m) + np.exp(z - m).sum(axis=-1, keepdims=True))) / beta
        # peak_cost = weights["peak"] * smooth_peak ** 2 / q_ptw.shape[-1]
        # peak_cost = np.broadcast_to(peak_cost, q_ptw.shape)
        q_prev = np.concatenate([q_prev[...,-1:], q_ptw[...,:-1]], axis=-1)
        smooth_cost = weights["smooth"] * (q_ptw - q_prev) ** 2
        flood_cost = weights["flood"] * q_w ** 2
        costs = np.stack([flat_cost, smooth_cost, flood_cost], axis=-1)
        return costs

    def _stage_cost_th(self, q_ptw, q_prev, q_w):
        obj = self.config["objective"]
        weights = obj["weights"]
        q_bar = th.concat([q_prev, q_ptw], dim=-1).mean(dim=-1, keepdim=True)
        flat_cost = weights["flat"] * (q_ptw - q_bar) ** 2
        # beta = float(obj.get("peak_smooth", 10.0))
        # zero = th.zeros_like(q_ptw[...,:1])
        # smooth_peak = th.logsumexp(th.concat([zero, beta * (q_ptw - q_bar)], dim=-1),
        #                            dim=-1, keepdim=True) / beta
        # peak_cost = weights["peak"] * smooth_peak ** 2 / q_ptw.shape[-1]
        # peak_cost = peak_cost.expand_as(q_ptw)
        q_prev = th.concat([q_prev[...,-1:], q_ptw[...,:-1]], dim=-1)
        smooth_cost = weights["smooth"] * (q_ptw - q_prev) ** 2
        flood_cost = weights["flood"] * q_w ** 2
        costs = th.stack([flat_cost, smooth_cost, flood_cost], dim=-1)
        return costs

    def objective(self, seq=False, keepdim=False):
        n = max(seq, 1) if seq else 1
        perfs = self.performance(seq = 2*n)
        q_ptw = np.asarray(perfs[-n:, 0]).squeeze()
        q_ptw = np.atleast_1d(q_ptw)
        q_prev = np.asarray(perfs[:n, 0]).squeeze()
        q_prev = np.atleast_1d(q_prev)
        q_w = self.flood(seq=n).squeeze().sum(axis=-1)
        q_w = np.atleast_1d(q_w)
        obj = self._stage_cost_np(q_ptw, q_prev, q_w)
        if seq:
            return obj.sum(axis=-1) if not keepdim else obj
        else:
            return obj.squeeze()

    def objective_pred(self, preds, states, settings, gamma=None, keepdim=False):
        preds, _ = preds
        state, _ = states
        nodes = self.elements["nodes"]
        ptw_idx = nodes.index(self.config["objective"]["ptw_outfall"])
        q_ptw = preds[..., ptw_idx, 1]
        q_w = preds[..., -1].sum(axis=-1)
        obj = self._stage_cost_np(q_ptw, state[..., ptw_idx, 1], q_w)
        gamma = np.ones(preds.shape[1]) if gamma is None else np.array(gamma, dtype=np.float32)
        obj = obj * gamma.reshape((1, -1, 1))
        return obj if keepdim else obj.sum(axis=-1)

    def objective_pred_th(self, preds, states, settings, gamma=None, keepdim=False):
        preds, _ = preds
        state, _ = states
        nodes = self.elements["nodes"]
        ptw_idx = nodes.index(self.config["objective"]["ptw_outfall"])
        q_ptw = preds[..., ptw_idx, 1]
        q_w = preds[..., -1].sum(dim=-1)
        obj = self._stage_cost_th(q_ptw, state[..., ptw_idx, 1], q_w)
        gamma = th.ones(preds.shape[1], device=obj.device) if gamma is None else th.tensor(gamma, device=obj.device)
        obj = obj * gamma.reshape((1, -1, 1))
        return obj if keepdim else obj.sum(dim=-1)

    def get_action_space(self, act="rand"):
        if act.startswith("conti"):
            return {k: (min(v), max(v)) for k, v in self.config["action_space"].items()}
        return self.config["action_space"].copy()

    def get_action_table(self, act="rand"):
        asp = self.get_action_space(act)
        actions = product(*[range(len(v)) for v in asp.values()])
        return {act: [v[a] for a, v in zip(act, asp.values())] for act in actions}

    def get_args(self, directed=False, length=0, order=1, graph_base=0, act=False, dec=False):
        args = super().get_args(directed, length, order, graph_base)
        for key in ["rainfall_timeseries", "rainfall_events"]:
            if not os.path.isfile(args["rainfall"][key]):
                args["rainfall"][key] = os.path.join(os.path.dirname(HERE), "config", args["rainfall"][key] + ".csv")
        args["area"] = args["is_storage"].astype(np.float32)
        act_edges = self.get_edge_list(list(self.config["action_space"].keys()))
        act_edges = [np.where((args["edges"] == act_edge).all(1))[0] for act_edge in act_edges]
        act_edges = [i for e in act_edges for i in e]
        args["act_edges"] = sorted(list(set(act_edges)), key=act_edges.index)
        if act and self.config["act"]:
            args["action_space"] = self.get_action_space(act)
            if not act.startswith("conti"):
                args["action_table"] = self.get_action_table(act)
                args["action_shape"] = np.array(list(args["action_table"].keys())).max(axis=0) + 1
            else:
                args["action_shape"] = len(args["action_space"])
            args["n_agents"] = 1
            args["observ_space"] = self.config["states"]
        return args

    def controller(self, mode="rand", state=None, setting=None):
        asp = self.get_action_space(mode)
        if mode.lower().startswith("rand"):
            return [table[np.random.randint(0, len(table))] for table in asp.values()]
        if mode.lower().startswith("conti"):
            return [np.random.uniform(min(table), max(table)) for table in asp.values()]
        if mode.lower() in ["off", "closed"]:
            return [table[0] for table in asp.values()]
        if mode.lower() in ["on", "open", "default"]:
            return [table[-1] for table in asp.values()]
        if mode.lower() == "safe":
            return setting
        raise AssertionError("Unknown controller %s" % str(mode))
