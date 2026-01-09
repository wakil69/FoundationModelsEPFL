from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from copy import deepcopy
from tta.base import TTAMethod

def disable_running_stats(model):
    """
    Disable BN running_mean/var and force batch statistics.
    BatchNorm computes statistics from the current test batch, which matches the actual distribution of the corrupted data
    input: model
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

def enable_bn_grads(model):
    """
    Function to selectively freezing/unfreezing parameters
    Enable gradients only for BN affine parameters: TENT (γ and β)
    """
    for p in model.parameters():
        p.requires_grad = False # freeze everything

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = True # unfreeze
            m.bias.requires_grad = True # unfreeze

def collect_bn_params(model):
    """
    Function that return BN affine parameters (m.weight: gamma, m.bias: beta) for the optimizer
    input: the model 
    output: the parameters
    """
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.weight is not None and m.weight.requires_grad:
                params.append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                params.append(m.bias)
    return params

def entropy_loss(logits):
    """
    Function to define TENT loss (entropy minimization)
    """
    p = F.softmax(logits, dim=1)
    return -(p * torch.log(p + 1e-12)).sum(dim=1).mean()

class Submission(TTAMethod):
    """
    TENT: Test-Time Entropy Minimization
    Updates ONLY BatchNorm affine parameters (gamma, beta).
    Uses batch statistics (no running stats).
    """

    def __init__(self, model, lr: float = 1e-3, momentum: float = 0.9, **kwargs):
        super().__init__(model)

        disable_running_stats(self.model) # Disable BN running_mean/var and force batch statistics
        enable_bn_grads(self.model) # freeze all parameteres but gamma and beta

        params = collect_bn_params(self.model) # collect BN paramas gammas and betas
        self.optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum) # optimizer for gamma and beta 
        self.model_state = deepcopy(self.model.state_dict())
        self.model.train() # Put the model in training mode

        # Add weak augmentation: During test-time, images are augmented with mild spatial transformations.
        # TENT becomes more stable when predictions are averaged over mild augmentations
        self.aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4)
        ])

    def reset(self):
        """
        Reset model to initial pretrained state, restore original weights
        """
        self.model.load_state_dict(self.model_state, strict=True)
        disable_running_stats(self.model)
        enable_bn_grads(self.model)

    def __call__(self, x):
        """
        Forward with augmentation consistency.
        """
        self.optimizer.zero_grad()
        
        # Original prediction
        logits = self.model(x)
        
        # Augmented prediction
        x_aug = self.aug(x)
        logits_aug = self.model(x_aug)
        
        # Combined loss: entropy + consistency
        loss = (entropy_loss(logits) + entropy_loss(logits_aug)) * 0.5
        
        loss.backward()
        self.optimizer.step()
        
        return logits  # Return original for evaluation
    
