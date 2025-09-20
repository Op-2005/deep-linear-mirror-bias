You are an expert ML research engineer. Upgrade the current codebase to a fully working, research-grade implementation for studying the implicit bias of Mirror Descent (MD) in deep linear networks (DLNs). Keep code clean, modular, reproducible, and aligned with theory.
Do not change the repo layout; instead, fill in TODOs and stubs with correct implementations.
1) Models (core/models.py)
Implement Linear(input_dim, output_dim=1, bias=True) as a plain linear classifier.
Implement DeepLinear(input_dim, hidden_dims: list[int], output_dim=1, bias=True) as a stack of nn.Linear layers without nonlinearities.
Add effective_weight(cache=True): multiply weights right-to-left into a single vector/matrix u; cache it and invalidate on .train() or weight changes.
Add get_layer_weights() to return only weight tensors (ignoring biases).
2) Potentials (core/potentials.py)
Each class must implement:
value(W) → scalar potential.
grad(W) → ∇φ(W) (mirror map).
grad_inv(Z) → (∇φ)⁻¹(Z) (inverse mirror map), ε-safe with clamp 1e-12.
Implement:
Quadratic: φ(W)=½‖W‖², grad=identity.
LayerScaledQuadratic(α): φ(W)=½α‖W‖², grad(W)=αW, grad_inv(Z)=Z/α.
LpPotential(p>1): φ(W)=Σ|Wᵢⱼ|ᵖ/p, grad(W)=sign(W)|W|^(p-1), grad_inv(Z)=sign(Z)|Z|^(1/(p-1)).
ScaledPotential: convex, magnitude-dependent (e.g. g(r)=r²/(1+r²)), with stable grad/grad_inv (Newton or clamp).
3) Optimizer (core/md_optimizer.py)
Implement MirrorDescentOptimizer with:
Maintained dual buffers Zℓ per layer.
Update rule per layer:
Zℓ ← Zℓ − lr * (Wℓ.grad + weight_decay*Wℓ)
If normalize_md=True, divide update by the dual norm of the gradient (for ℓₚ, use q with 1/p+1/q=1).
Wℓ.data ← potentials[ℓ].grad_inv(Zℓ)
Add dual_clip (optional clamp), state_dict, load_state_dict, set_lr.
4) Data (data/synthetic.py, data/mnist.py)
Synthetic:
generate_gaussian_data(n, d, separation, noise, seed) → two Gaussians with means ±μ, ‖μ‖=separation, covariance σ²I, labels ±1.
standardize(X) and train_val_split.
MNIST:
load_mnist_binary(d1,d2, flatten=True, pca_components=None, standardize=True, seed=0).
Return torch tensors (X_train,y_train,X_test,y_test) with labels ±1.
5) Baselines (eval/baselines.py)
svm_l2(X,y): sklearn LinearSVC(C=1e6, loss='hinge'), return normalized weight vector.
logreg_gd(X,y): sklearn LogisticRegression(penalty='none', solver='saga'), return normalized weight.
Leave clear TODO stub for generalized p-margin (cvxpy).
6) Metrics (eval/metrics.py)
Implement:
margin(u,X,y): min_i yᵢ(uᵀxᵢ)/‖u‖.
angle(u,v): degrees, safe clamp.
angles_to_baselines(u,X,y,baselines): angles to svm/logreg (and linear-MD if provided).
layer_alignment(W_list): cosine similarity between top singular vectors across layers.
norm_balance(W_list): max/min Frobenius norms across layers.
Stub for ntk_drift with docstring.
7) Training Script (runs/scripts/train_synth.py)
CLI args: --potential {quadratic,lp,layer_scaled,scaled}, --p, --L, --input_dim, --n_samples, --separation, --noise, --lr, --epochs, --seed, --normalize_md.
Build dataset, train both Linear MD and DeepLinear MD with same potential.
Compute effective weight u_deep each epoch.
Baselines: fit SVM/logreg once.
Log: margin, angles to baselines, norm balance, layer alignment.
Save: decision boundary plot (if 2D) and JSON with final metrics.

8) README Update
Add:
Quick project description (implicit bias of mirror descent in deep linear nets).
Setup instructions (pip install -r requirements.txt).
“First run” example:
python -m runs.scripts.train_synth --potential quadratic --L 2 --input_dim 2 --n_samples 512 --separation 3.0 --noise 0.2 --lr 0.1 --epochs 1000 --seed 0
Mention results saved to reports/figs and reports/tables.
9) Testing & Stability
Add doctest checks in potentials.py: verify grad_inv(grad(W)) ≈ W for random W.
Clamp in grad_inv to avoid NaNs.
Seed torch, numpy, sklearn for reproducibility.
CPU-only safe.

Key Design Notes
DeepLinear.effective_weight caching is critical for efficient angle/margin tracking (pattern from lucidrains’ repo).
Normalized MD option allows reproducing p-GD findings (generalized margin bias, faster convergence).
ℓₚ potentials use elementwise dual/inverse mapping (cheap, exact).
Layer-scaled Φ enables depth × geometry experiments.