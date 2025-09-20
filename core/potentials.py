"""
Potential Functions for Mirror Descent Optimization.

This module implements various potential functions that define the geometry
of the dual space in Mirror Descent. Different potentials lead to different
implicit biases in the optimization process.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from abc import ABC, abstractmethod


def test_potential_consistency(potential, W: torch.Tensor, tolerance: float = 1e-6) -> bool:
    """
    Test that grad_inv(grad(W)) ≈ W for a given potential function.
    
    This is a critical consistency check for potential functions.
    
    Args:
        potential: Potential function to test
        W: Test weight tensor
        tolerance: Tolerance for the test
        
    Returns:
        True if test passes, False otherwise
        
    >>> import torch
    >>> from core.potentials import QuadraticPotential, LpPotential
    >>> torch.manual_seed(42)
    >>> W = torch.randn(3, 4)
    >>> 
    >>> # Test quadratic potential
    >>> quad_pot = QuadraticPotential()
    >>> test_potential_consistency(quad_pot, W)
    True
    >>> 
    >>> # Test Lp potential
    >>> lp_pot = LpPotential(p=2.5)
    >>> test_potential_consistency(lp_pot, W)
    True
    """
    try:
        # Compute gradient and inverse gradient
        grad_W = potential.grad(W)
        recovered_W = potential.grad_inv(grad_W)
        
        # Check if recovery is close to original
        diff = torch.norm(W - recovered_W)
        return diff.item() < tolerance
    except Exception:
        return False


class PotentialFunction(ABC):
    """
    Abstract base class for potential functions in Mirror Descent.
    
    Each class must implement:
    - value(W) → scalar potential
    - grad(W) → ∇φ(W) (mirror map)  
    - grad_inv(Z) → (∇φ)⁻¹(Z) (inverse mirror map), ε-safe with clamp 1e-12
    """
    
    @abstractmethod
    def value(self, W: torch.Tensor) -> torch.Tensor:
        """Compute potential value φ(W)."""
        pass
        
    @abstractmethod
    def grad(self, W: torch.Tensor) -> torch.Tensor:
        """Compute gradient ∇φ(W) (mirror map)."""
        pass
        
    @abstractmethod
    def grad_inv(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute inverse gradient (∇φ)⁻¹(Z), ε-safe with clamp 1e-12."""
        pass


class QuadraticPotential(PotentialFunction):
    """
    Quadratic potential function: φ(W) = ½‖W‖².
    
    This corresponds to standard gradient descent in the primal space.
    """
    
    def __init__(self):
        """Initialize quadratic potential."""
        pass
        
    def value(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute quadratic potential: ½‖W‖².
        
        Args:
            W: Weight tensor
            
        Returns:
            Potential value
        """
        return 0.5 * torch.sum(W ** 2)
        
    def grad(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient: W (identity map).
        
        Args:
            W: Weight tensor
            
        Returns:
            Gradient of potential
        """
        return W
        
    def grad_inv(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse gradient: Z (identity map).
        
        Args:
            Z: Dual variable
            
        Returns:
            Inverse gradient
        """
        return Z


class LayerScaledQuadratic(PotentialFunction):
    """
    Layer-scaled quadratic potential: φ(W) = ½α‖W‖².
    
    Args:
        alpha: Scaling parameter α > 0
    """
    
    def __init__(self, alpha: float):
        """
        Initialize layer-scaled quadratic potential.
        
        Args:
            alpha: Scaling parameter α > 0
        """
        assert alpha > 0, "Alpha must be positive"
        self.alpha = alpha
        
    def value(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute layer-scaled quadratic potential: ½α‖W‖².
        
        Args:
            W: Weight tensor
            
        Returns:
            Potential value
        """
        return 0.5 * self.alpha * torch.sum(W ** 2)
        
    def grad(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient: αW.
        
        Args:
            W: Weight tensor
            
        Returns:
            Gradient of potential
        """
        return self.alpha * W
        
    def grad_inv(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse gradient: Z/α.
        
        Args:
            Z: Dual variable
            
        Returns:
            Inverse gradient
        """
        return Z / self.alpha


class LpPotential(PotentialFunction):
    """
    Lp potential function: φ(W) = Σ|Wᵢⱼ|ᵖ/p.
    
    For p>1, this gives elementwise dual/inverse mapping.
    """
    
    def __init__(self, p: float):
        """
        Initialize Lp potential.
        
        Args:
            p: Order of the Lp norm (must be > 1)
        """
        assert p > 1, "p must be greater than 1"
        self.p = p
        self.q = p / (p - 1)  # Dual exponent: 1/p + 1/q = 1
        
    def value(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute Lp potential: Σ|Wᵢⱼ|ᵖ/p.
        
        Args:
            W: Weight tensor
            
        Returns:
            Potential value
        """
        return torch.sum(torch.abs(W) ** self.p) / self.p
        
    def grad(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient: sign(W)|W|^(p-1).
        
        Args:
            W: Weight tensor
            
        Returns:
            Gradient of potential
        """
        return torch.sign(W) * torch.abs(W) ** (self.p - 1)
        
    def grad_inv(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse gradient: sign(Z)|Z|^(1/(p-1)).
        
        Args:
            Z: Dual variable
            
        Returns:
            Inverse gradient
        """
        # Clamp to avoid numerical issues
        Z_clamped = torch.clamp(torch.abs(Z), min=1e-12)
        return torch.sign(Z) * Z_clamped ** (1 / (self.p - 1))


class ScaledPotential(PotentialFunction):
    """
    Scaled potential function with magnitude-dependent scaling.
    
    Uses g(r) = r²/(1+r²) for convex, magnitude-dependent scaling.
    """
    
    def __init__(self, epsilon: float = 1e-12):
        """
        Initialize scaled potential.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        self.epsilon = epsilon
        
    def value(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled potential with magnitude-dependent scaling.
        
        Args:
            W: Weight tensor
            
        Returns:
            Potential value
        """
        # Magnitude-dependent scaling: g(r) = r²/(1+r²)
        norms_sq = torch.sum(W ** 2, dim=-1, keepdim=True)
        scaling = norms_sq / (1 + norms_sq)
        return torch.sum(scaling * norms_sq)
        
    def grad(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient with magnitude-dependent scaling.
        
        Args:
            W: Weight tensor
            
        Returns:
            Gradient of potential
        """
        norms_sq = torch.sum(W ** 2, dim=-1, keepdim=True)
        # Derivative of g(r)r² where g(r) = r²/(1+r²)
        # d/dr[g(r)r²] = d/dr[r⁴/(1+r²)] = [4r³(1+r²) - r⁴(2r)]/(1+r²)²
        # = r³[4(1+r²) - 2r²]/(1+r²)² = r³[4 + 2r²]/(1+r²)²
        denominator = (1 + norms_sq) ** 2
        numerator = norms_sq * (4 + 2 * norms_sq)
        scaling = numerator / torch.clamp(denominator, min=self.epsilon)
        return W * scaling
        
    def grad_inv(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse gradient using Newton's method with clamping.
        
        Args:
            Z: Dual variable
            
        Returns:
            Inverse gradient
        """
        # For complex inverse, use Newton's method with clamping
        # Start with Z as initial guess
        W = Z.clone()
        
        for _ in range(10):  # Max 10 Newton iterations
            grad_W = self.grad(W)
            diff = grad_W - Z
            if torch.norm(diff) < 1e-6:
                break
                
            # Approximate Hessian diagonal for Newton step
            norms_sq = torch.sum(W ** 2, dim=-1, keepdim=True)
            hessian_diag = (4 + 2 * norms_sq) / (1 + norms_sq) ** 2
            hessian_diag = torch.clamp(hessian_diag, min=self.epsilon)
            
            # Newton step
            W = W - diff / hessian_diag
            
        return torch.clamp(W, min=-1e6, max=1e6)


class EntropyPotential(PotentialFunction):
    """
    Entropy potential function: φ(W) = Σ Wᵢ log(Wᵢ).
    
    Commonly used in online learning and game theory applications.
    """
    
    def __init__(self, epsilon: float = 1e-12):
        """
        Initialize entropy potential.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        self.epsilon = epsilon
        
    def value(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy potential.
        
        Args:
            W: Input tensor (should be non-negative)
            
        Returns:
            Potential value
        """
        # Ensure non-negative for entropy
        W_pos = torch.clamp(W, min=self.epsilon)
        return torch.sum(W_pos * torch.log(W_pos))
        
    def grad(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of entropy potential: 1 + log(W).
        
        Args:
            W: Input tensor
            
        Returns:
            Gradient of potential
        """
        W_clamped = torch.clamp(W, min=self.epsilon)
        return 1 + torch.log(W_clamped)
        
    def grad_inv(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse gradient: exp(Z - 1).
        
        Args:
            Z: Dual variable
            
        Returns:
            Inverse gradient
        """
        result = torch.exp(Z - 1)
        return torch.clamp(result, min=self.epsilon, max=1e6)


def run_potential_tests():
    """
    Run comprehensive tests on all potential functions.
    
    This function tests the consistency of grad_inv(grad(W)) ≈ W
    for all potential functions with random weight tensors.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test different weight tensor shapes
    test_shapes = [(3, 4), (10, 5), (2, 2, 3)]
    
    # Test different potential functions
    potentials_to_test = [
        QuadraticPotential(),
        LpPotential(p=1.5),
        LpPotential(p=2.0),
        LpPotential(p=3.0),
        LayerScaledQuadratic(alpha=0.5),
        LayerScaledQuadratic(alpha=2.0),
        ScaledPotential(),
        EntropyPotential()
    ]
    
    print("Running potential function consistency tests...")
    
    all_passed = True
    
    for potential in potentials_to_test:
        potential_name = type(potential).__name__
        print(f"\nTesting {potential_name}:")
        
        for shape in test_shapes:
            W = torch.randn(*shape)
            
            # Skip entropy potential for negative weights
            if isinstance(potential, EntropyPotential):
                W = torch.abs(W) + 1e-6
            
            passed = test_potential_consistency(potential, W)
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Shape {shape}: {status}")
            
            if not passed:
                all_passed = False
    
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return all_passed


if __name__ == "__main__":
    import numpy as np
    run_potential_tests()
