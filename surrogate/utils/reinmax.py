# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Straight-Through Estimator (STE) for discrete action spaces in MPC.
# Forked from https://github.com/microsoft/ReinMax
import torch

class ReinMax_Auto(torch.autograd.Function):
    """
    `torch.autograd.Function` implementation of the ReinMax gradient estimator.
    """
    
    @staticmethod
    def forward(
        ctx, 
        logits: torch.Tensor, 
        tau: torch.Tensor,
    ):
        y_soft = logits.softmax(dim=-1)
        sample = torch.argmax(
            y_soft,
            dim=-1,
            keepdims=True,
        )
        one_hot_sample = torch.zeros_like(
            y_soft, 
            memory_format=torch.legacy_contiguous_format
        ).scatter_(-1, sample, 1.0)
        ctx.save_for_backward(one_hot_sample, logits, y_soft, tau)
        return one_hot_sample, y_soft

    @staticmethod
    def backward(
        ctx, 
        grad_at_sample: torch.Tensor, 
        grad_at_p: torch.Tensor,
    ):
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors
        
        shifted_y_soft = .5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)
        
        grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)
        
        grad_at_input = grad_at_input_0 + grad_at_input_1
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None

def reinmax(
        logits: torch.Tensor, 
        tau: float, 
        hard: bool = True,
    ):
    r"""
    ReinMax Gradient Approximation.

    Parameters
    ---------- 
    logits: ``torch.Tensor``, required.
        The input Tensor for the softmax. Note that the softmax operation would be conducted along the last dimension. 
    tau: ``float``, required. 
        The temperature hyper-parameter. 

    Returns
    -------
    sample: ``torch.Tensor``.
        The one-hot sample generated from ``multinomial(softmax(logits))``. 
    p: ``torch.Tensor``.
        The output of the softmax function, i.e., ``softmax(logits)``. 
    """
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMax_Auto.apply(logits, logits.new_empty(1).fill_(tau))
    return grad_sample.view(shape), y_soft.view(shape)