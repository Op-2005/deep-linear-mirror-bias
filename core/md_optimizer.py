"""
Mirror Descent Optimizer Implementation.

This module implements various Mirror Descent algorithms for training deep linear networks.
The focus is on understanding how different potential functions (quadratic, lp-norms, etc.)
affect the implicit bias of the optimization process.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional, Callable, List, Union
from .potentials import PotentialFunction


class MirrorDescentOptimizer(optim.Optimizer):
    """
    Mirror Descent optimizer implementation.
    
    Implements the general Mirror Descent algorithm with configurable potential functions.
    Maintains dual buffers Zℓ per layer and applies the update rule:
    Zℓ ← Zℓ − lr * (Wℓ.grad + weight_decay*Wℓ)
    Wℓ.data ← potentials[ℓ].grad_inv(Zℓ)
    """
    
    def __init__(self, 
                 params,
                 potentials: Union[PotentialFunction, List[PotentialFunction]],
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.0,
                 normalize_md: bool = False,
                 dual_clip: Optional[float] = None):
        """
        Initialize Mirror Descent optimizer.
        
        Args:
            params: Model parameters to optimize (iterable of parameter groups)
            potentials: Potential function(s) defining the geometry
            learning_rate: Learning rate for the optimization
            weight_decay: Weight decay coefficient
            normalize_md: Whether to normalize by dual norm of gradient
            dual_clip: Optional clipping value for dual variables
        """
        defaults = dict(
            lr=learning_rate,
            weight_decay=weight_decay,
            normalize_md=normalize_md,
            dual_clip=dual_clip
        )
        super().__init__(params, defaults)
        
        # Handle potential functions
        if isinstance(potentials, PotentialFunction):
            self.potentials = [potentials] * len(self.param_groups)
        else:
            assert len(potentials) == len(self.param_groups), \
                "Number of potentials must match number of parameter groups"
            self.potentials = potentials
        
        # Initialize dual variables Z for each parameter group
        self.dual_vars = []
        for group in self.param_groups:
            group_dual_vars = []
            for p in group['params']:
                if p.requires_grad:
                    # Initialize dual variables to potential gradient of initial weights
                    dual_var = self.potentials[len(self.dual_vars)].grad(p.data)
                    group_dual_vars.append(dual_var)
                else:
                    group_dual_vars.append(None)
            self.dual_vars.append(group_dual_vars)
        
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform one optimization step using Mirror Descent.
        
        Args:
            closure: Optional closure for re-evaluating the model
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_idx, group in enumerate(self.param_groups):
            lr = group['lr']
            weight_decay = group['weight_decay']
            normalize_md = group['normalize_md']
            dual_clip = group['dual_clip']
            potential = self.potentials[group_idx]
            
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                    
                if p.grad.is_sparse:
                    raise RuntimeError('MirrorDescentOptimizer does not support sparse gradients')
                
                grad = p.grad.data
                param = p.data
                dual_var = self.dual_vars[group_idx][param_idx]
                
                if dual_var is None:
                    continue
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)
                
                # Normalize by dual norm if requested
                if normalize_md:
                    if hasattr(potential, 'q'):  # Lp potential
                        q = potential.q
                        dual_norm = torch.norm(grad, p=q)
                    else:  # Default to L2 norm
                        dual_norm = torch.norm(grad, p=2)
                    if dual_norm > 1e-8:
                        grad = grad / dual_norm
                
                # Update dual variables: Zℓ ← Zℓ − lr * grad
                dual_var.add_(grad, alpha=-lr)
                
                # Apply dual clipping if specified
                if dual_clip is not None:
                    torch.clamp_(dual_var, min=-dual_clip, max=dual_clip)
                
                # Update primal variables: Wℓ ← ∇φ⁻¹(Zℓ)
                p.data.copy_(potential.grad_inv(dual_var))
        
        return loss
        
    def get_dual_parameters(self) -> Dict[str, List[torch.Tensor]]:
        """
        Get dual space parameters for analysis.
        
        Returns:
            Dictionary of dual parameters by group
        """
        dual_params = {}
        for group_idx, group in enumerate(self.param_groups):
            group_name = f"group_{group_idx}"
            dual_params[group_name] = []
            for param_idx, p in enumerate(group['params']):
                if p.requires_grad:
                    dual_params[group_name].append(self.dual_vars[group_idx][param_idx].clone())
        return dual_params
    
    def set_lr(self, learning_rate: float):
        """
        Set learning rate for all parameter groups.
        
        Args:
            learning_rate: New learning rate
        """
        for group in self.param_groups:
            group['lr'] = learning_rate
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return the state of the optimizer as a dict.
        """
        state = super().state_dict()
        state['dual_vars'] = []
        for group_dual_vars in self.dual_vars:
            group_state = []
            for dual_var in group_dual_vars:
                if dual_var is not None:
                    group_state.append(dual_var.clone())
                else:
                    group_state.append(None)
            state['dual_vars'].append(group_state)
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load the state of the optimizer from a dict.
        """
        super().load_state_dict(state_dict)
        if 'dual_vars' in state_dict:
            self.dual_vars = []
            for group_dual_vars in state_dict['dual_vars']:
                group_vars = []
                for dual_var in group_dual_vars:
                    if dual_var is not None:
                        group_vars.append(dual_var.clone())
                    else:
                        group_vars.append(None)
                self.dual_vars.append(group_vars)


class AdaptiveMirrorDescent(MirrorDescentOptimizer):
    """
    Adaptive Mirror Descent with automatic step size selection.
    
    Extends the basic Mirror Descent with adaptive learning rates
    based on the potential function geometry.
    """
    
    def __init__(self, 
                 params,
                 potentials: Union[PotentialFunction, List[PotentialFunction]],
                 initial_lr: float = 1e-3,
                 adaptive: bool = True,
                 **kwargs):
        """
        Initialize adaptive Mirror Descent optimizer.
        
        Args:
            params: Model parameters to optimize
            potentials: Potential function(s) defining the geometry
            initial_lr: Initial learning rate
            adaptive: Whether to use adaptive step sizes
            **kwargs: Additional arguments for base MirrorDescentOptimizer
        """
        super().__init__(params, potentials, initial_lr, **kwargs)
        self.adaptive = adaptive
        self.initial_lr = initial_lr
        
    def _compute_adaptive_step_size(self, grad_norm: float, group_idx: int) -> float:
        """
        Compute adaptive step size based on gradient norm and potential geometry.
        
        Args:
            grad_norm: Current gradient norm
            group_idx: Index of parameter group
            
        Returns:
            Adaptive step size
        """
        if not self.adaptive:
            return self.param_groups[group_idx]['lr']
        
        # Simple adaptive scheme: reduce step size for large gradients
        # More sophisticated schemes could be implemented based on potential geometry
        if grad_norm > 1.0:
            return self.initial_lr / (1 + 0.1 * grad_norm)
        else:
            return self.initial_lr
            
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform one adaptive optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group_idx, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            normalize_md = group['normalize_md']
            dual_clip = group['dual_clip']
            potential = self.potentials[group_idx]
            
            for param_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                    
                if p.grad.is_sparse:
                    raise RuntimeError('AdaptiveMirrorDescent does not support sparse gradients')
                
                grad = p.grad.data
                param = p.data
                dual_var = self.dual_vars[group_idx][param_idx]
                
                if dual_var is None:
                    continue
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)
                
                # Compute adaptive learning rate
                grad_norm = torch.norm(grad)
                lr = self._compute_adaptive_step_size(grad_norm, group_idx)
                
                # Normalize by dual norm if requested
                if normalize_md:
                    if hasattr(potential, 'q'):  # Lp potential
                        q = potential.q
                        dual_norm = torch.norm(grad, p=q)
                    else:  # Default to L2 norm
                        dual_norm = torch.norm(grad, p=2)
                    if dual_norm > 1e-8:
                        grad = grad / dual_norm
                
                # Update dual variables: Zℓ ← Zℓ − lr * grad
                dual_var.add_(grad, alpha=-lr)
                
                # Apply dual clipping if specified
                if dual_clip is not None:
                    torch.clamp_(dual_var, min=-dual_clip, max=dual_clip)
                
                # Update primal variables: Wℓ ← ∇φ⁻¹(Zℓ)
                p.data.copy_(potential.grad_inv(dual_var))
        
        return loss
