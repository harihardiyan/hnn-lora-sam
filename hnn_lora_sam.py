
# ================== IMPORTS ==================
import jax
import jax.numpy as jnp
from jax import random, grad, value_and_grad, jit, lax
import optax
from typing import Dict

tree_map = jax.tree_util.tree_map

# ================== PHYSICS: TRUE HAMILTONIAN ==================

ALPHA = 1.0
BETA = 0.2
DT = 0.05

def true_hamiltonian(q, p, alpha=ALPHA, beta=BETA):
    return 0.5 * p**2 + 0.5 * alpha * q**2 + (beta / 4.0) * q**4

def hamiltonian_dynamics(x, dt=DT):
    q, p = x[..., 0], x[..., 1]
    dqdt = p
    dpdt = -ALPHA * q - BETA * q**3
    q_next = q + dt * dqdt
    p_next = p + dt * dpdt
    return jnp.stack([q_next, p_next], axis=-1)

def rollout_hamiltonian(key, n_trajectories=128, T=128, dt=DT):
    k_init, = random.split(key, 1)
    x0 = random.uniform(k_init, (n_trajectories, 2), minval=-2.0, maxval=2.0)

    def step_fn(carry, t):
        x = carry
        x_next = hamiltonian_dynamics(x, dt=dt)
        return x_next, (x, x_next)

    _, (xs, xs_next) = lax.scan(step_fn, x0, jnp.arange(T))
    xs = xs.reshape(T * n_trajectories, 2)
    xs_next = xs_next.reshape(T * n_trajectories, 2)
    return xs, xs_next

# ================== HNN MODEL (LoRA ON MULTIPLE LAYERS) ==================

def init_hnn_params(key, in_dim, hidden_dim, lora_rank, scale=0.02):
    k_in_base, k_in_A, k_in_B, k_hid_base, k_hid_A, k_hid_B, k_out_base, k_out_A, k_out_B, k_bhid, k_bout = random.split(key, 11)

    base_W_in = random.normal(k_in_base, (in_dim, hidden_dim)) * (1.0 / jnp.sqrt(in_dim))
    lora_A_in = random.normal(k_in_A, (in_dim, lora_rank)) * scale
    lora_B_in = jnp.zeros((lora_rank, hidden_dim))

    base_W_hid = random.normal(k_hid_base, (hidden_dim, hidden_dim)) * (1.0 / jnp.sqrt(hidden_dim))
    lora_A_hid = random.normal(k_hid_A, (hidden_dim, lora_rank)) * scale
    lora_B_hid = jnp.zeros((lora_rank, hidden_dim))

    base_W_out = random.normal(k_out_base, (hidden_dim, 1)) * (1.0 / jnp.sqrt(hidden_dim))
    lora_A_out = random.normal(k_out_A, (hidden_dim, lora_rank)) * scale
    lora_B_out = jnp.zeros((lora_rank, 1))

    b_hid = jnp.zeros((hidden_dim,))
    b_out = jnp.zeros((1,))

    return {
        "base_W_in": base_W_in,
        "lora_A_in": lora_A_in,
        "lora_B_in": lora_B_in,
        "base_W_hid": base_W_hid,
        "lora_A_hid": lora_A_hid,
        "lora_B_hid": lora_B_hid,
        "base_W_out": base_W_out,
        "lora_A_out": lora_A_out,
        "lora_B_out": lora_B_out,
        "b_hid": b_hid,
        "b_out": b_out,
    }

def apply_lora_linear(x, base_W, A, B, alpha=1.0):
    r = A.shape[1]
    base = x @ base_W
    lora = x @ A @ B
    return base + (alpha / r) * lora

def hnn_forward(params, x, alpha=1.0):
    h = apply_lora_linear(x, params["base_W_in"], params["lora_A_in"], params["lora_B_in"], alpha=alpha)
    h = jax.nn.tanh(h)
    h = apply_lora_linear(h, params["base_W_hid"], params["lora_A_hid"], params["lora_B_hid"], alpha=alpha)
    h = jax.nn.tanh(h + params["b_hid"])
    H = apply_lora_linear(h, params["base_W_out"], params["lora_A_out"], params["lora_B_out"], alpha=alpha)
    H = H + params["b_out"]
    return H.squeeze(-1)

# --------- HNN STEP WITH SYMPLECTIC EULER ---------

def hnn_step(params, x, dt=DT):
    def H_fn(x_single):
        return hnn_forward(params, x_single[None, :])[0]

    dHdx = jax.vmap(grad(H_fn))(x)
    dHdq = dHdx[:, 0]
    dHdp = dHdx[:, 1]

    q = x[:, 0]
    p = x[:, 1]

    p_next = p - dt * dHdq
    q_next = q + dt * dHdp

    return jnp.stack([q_next, p_next], axis=-1)

# ================== LOSS: 1-STEP + LONG ROLLOUT ==================

def one_step_loss(params, batch):
    x, y = batch
    preds = hnn_step(params, x)
    mse = jnp.mean((preds - y) ** 2)
    return mse

def long_rollout_loss(params, x0, y_seq, horizon, teacher_forcing_ratio, key):
    def step_fn(carry, t):
        x_curr, key = carry
        key, k_tf = random.split(key)
        use_teacher = random.bernoulli(k_tf, p=teacher_forcing_ratio)
        x_true = y_seq[t]
        x_model = hnn_step(params, x_curr)
        x_next = jnp.where(use_teacher, x_true, x_model)
        return (x_next, key), (x_model, x_true)

    (_, _), (xs_model, xs_true) = lax.scan(
        step_fn,
        (x0, key),
        jnp.arange(horizon)
    )
    mse = jnp.mean((xs_model - xs_true) ** 2)
    return mse

def compute_total_loss(params, batch, long_rollout_cfg, key):
    x, y = batch
    mse_1 = one_step_loss(params, batch)
    total = mse_1

    if long_rollout_cfg["enabled"]:
        H = long_rollout_cfg["horizon"]
        key, k_roll = random.split(key)

        def dyn_step(carry, t):
            x_curr = carry
            x_next = hamiltonian_dynamics(x_curr)
            return x_next, x_next

        _, y_seq = lax.scan(dyn_step, x, jnp.arange(H))
        long_mse = long_rollout_loss(
            params,
            x0=x,
            y_seq=y_seq,
            horizon=H,
            teacher_forcing_ratio=long_rollout_cfg["teacher_forcing_ratio"],
            key=k_roll,
        )
        total = total + long_rollout_cfg["weight"] * long_mse
    else:
        long_mse = 0.0

    return total, (mse_1, long_mse)

# ================== HVP & LANCZOS (CURVATURE) ==================

def pytree_norm(tree):
    return jnp.sqrt(sum([jnp.sum(x**2) for x in jax.tree_util.tree_leaves(tree)]))

def pytree_scale(tree, scalar):
    return tree_map(lambda x: x * scalar, tree)

def pytree_sub(a, b):
    return tree_map(lambda x, y: x - y, a, b)

def hvp(loss_fn, params, batch, vec_params, long_rollout_cfg, key):
    def loss_wrapped(p):
        total, _ = loss_fn(p, batch, long_rollout_cfg, key)
        return total
    grad_fn = grad(loss_wrapped)
    hv, _ = jax.jvp(grad_fn, (params,), (vec_params,))
    return hv

def init_random_vec_like_params(key, params):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = random.split(key, len(leaves))
    v_leaves = [random.normal(k, p.shape) for p, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, v_leaves)

def lanczos_top_eig(loss_fn, params, batch, key, long_rollout_cfg, max_iters=20):
    v0 = init_random_vec_like_params(key, params)
    v0_norm = pytree_norm(v0)
    v = pytree_scale(v0, 1.0 / (v0_norm + 1e-8))
    v_prev = tree_map(jnp.zeros_like, v)

    alphas = jnp.zeros((max_iters,))
    betas = jnp.zeros((max_iters,))

    def body_fun(i, carry):
        v, v_prev, alphas, betas = carry
        Hv = hvp(loss_fn, params, batch, v, long_rollout_cfg, key)

        alpha = sum([
            jnp.vdot(x, y)
            for x, y in zip(
                jax.tree_util.tree_leaves(Hv),
                jax.tree_util.tree_leaves(v)
            )
        ])
        alphas = alphas.at[i].set(alpha)

        w = pytree_sub(Hv, pytree_scale(v, alpha))
        beta_prev = jnp.where(i == 0, 0.0, betas[i-1])
        w = pytree_sub(w, pytree_scale(v_prev, beta_prev))

        beta = pytree_norm(w)
        betas = betas.at[i].set(beta)

        v_next = pytree_scale(w, 1.0 / (beta + 1e-8))
        return (v_next, v, alphas, betas)

    v_final, v_prev_final, alphas, betas = lax.fori_loop(
        0, max_iters, body_fun, (v, v_prev, alphas, betas)
    )

    k = max_iters
    T = jnp.diag(alphas)
    off = betas[:-1]
    T = T.at[jnp.arange(k-1), jnp.arange(1, k)].set(off)
    T = T.at[jnp.arange(1, k), jnp.arange(k-1)].set(off)

    eigvals = jnp.linalg.eigvalsh(T)
    return eigvals[-1]

def curvature_signal(params, batch, key, long_rollout_cfg):
    return lanczos_top_eig(compute_total_loss, params, batch, key, long_rollout_cfg, max_iters=20)

# ================== MASKS & MASKED UPDATE ==================

def make_lora_mask(params: Dict[str, jnp.ndarray]):
    return {
        "base_W_in": False,
        "lora_A_in": True,
        "lora_B_in": True,
        "base_W_hid": False,
        "lora_A_hid": True,
        "lora_B_hid": True,
        "base_W_out": False,
        "lora_A_out": True,
        "lora_B_out": True,
        "b_hid": False,
        "b_out": False,
    }

def make_train_mask(params: Dict[str, jnp.ndarray]):
    return {
        "base_W_in": True,
        "lora_A_in": True,
        "lora_B_in": True,
        "base_W_hid": True,
        "lora_A_hid": True,
        "lora_B_hid": True,
        "base_W_out": True,
        "lora_A_out": True,
        "lora_B_out": True,
        "b_hid": True,
        "b_out": True,
    }

def masked_update(grads, params, optimizer, opt_state):
    train_mask = make_train_mask(params)
    grads_masked = tree_map(
        lambda g, m: g if m else jnp.zeros_like(g),
        grads, train_mask
    )
    updates, new_opt_state = optimizer.update(grads_masked, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

# ================== SAM + DYNAMIC ρ ==================

def sam_step(params, opt_state, batch, optimizer, rho, long_rollout_cfg, key):
    lora_mask = make_lora_mask(params)

    def loss_fn(p, b, cfg, k):
        total, _ = compute_total_loss(p, b, cfg, k)
        return total

    loss, grads = value_and_grad(loss_fn)(params, batch, long_rollout_cfg, key)

    lora_grads = tree_map(
        lambda g, m: g if m else jnp.zeros_like(g),
        grads, lora_mask
    )
    grad_norm = jnp.sqrt(
        sum([jnp.sum(g**2) for g in jax.tree_util.tree_leaves(lora_grads)])
    )
    scale = jnp.where(grad_norm > 1e-12, rho / (grad_norm + 1e-8), 0.0)

    def perturb(p, g, m):
        return jnp.where(m, p + scale * g, p)

    params_perturbed = tree_map(perturb, params, lora_grads, lora_mask)

    loss2, grads2 = value_and_grad(loss_fn)(params_perturbed, batch, long_rollout_cfg, key)
    new_params, new_opt_state = masked_update(grads2, params, optimizer, opt_state)

    return new_params, new_opt_state, loss2, grad_norm

def make_train_step_sam(optimizer, long_rollout_cfg):
    @jit
    def train_step_sam(params, opt_state, batch, rho, key):
        return sam_step(params, opt_state, batch, optimizer, rho, long_rollout_cfg, key)
    return train_step_sam

def make_train_step_standard(optimizer, long_rollout_cfg):
    @jit
    def train_step_standard(params, opt_state, batch, key):
        def loss_fn(p, b, cfg, k):
            total, _ = compute_total_loss(p, b, cfg, k)
            return total
        loss, grads = value_and_grad(loss_fn)(params, batch, long_rollout_cfg, key)
        new_params, new_opt_state = masked_update(grads, params, optimizer, opt_state)
        return new_params, new_opt_state, loss
    return train_step_standard

# ================== DATASET & BATCHING ==================

def make_hamiltonian_dataset(key, n_trajectories=256, T=128):
    X, Y = rollout_hamiltonian(key, n_trajectories=n_trajectories, T=T, dt=DT)
    return X, Y

def make_batches(X, Y, batch_size=128):
    n = X.shape[0]
    n_batches = n // batch_size
    X = X[:n_batches * batch_size]
    Y = Y[:n_batches * batch_size]
    Xb = X.reshape(n_batches, batch_size, -1)
    Yb = Y.reshape(n_batches, batch_size, -1)
    return list(zip(Xb, Yb))

# ================== RHO SCHEDULE ==================

def rho_schedule(base_rho, curvature, ref=0.01, max_scale=5.0):
    ratio = curvature / (ref + 1e-8)
    scale = 1.0 + 0.5 * (ratio - 1.0)
    scale = jnp.clip(scale, 0.5, max_scale)
    return float(base_rho * scale)

# ================== TRAIN LOOP ==================

def train(
    rng_key,
    epochs=5,
    batch_size=128,
    hidden_dim=512,
    lora_rank=32,
    base_lr=1e-3,
    use_sam=True,
):
    key_data, key_params, key_loop = random.split(rng_key, 3)
    X, Y = make_hamiltonian_dataset(key_data, n_trajectories=256, T=128)
    in_dim = X.shape[-1]

    batches = make_batches(X, Y, batch_size=batch_size)

    params = init_hnn_params(key_params, in_dim, hidden_dim, lora_rank)
    optimizer = optax.adamw(base_lr, weight_decay=1e-2)
    opt_state = optimizer.init(params)

    long_rollout_cfg = {
        "enabled": True,
        "horizon": 20,
        "teacher_forcing_ratio": 0.5,
        "weight": 0.3,
    }

    train_step_sam = make_train_step_sam(optimizer, long_rollout_cfg)
    train_step_standard = make_train_step_standard(optimizer, long_rollout_cfg)

    rng = key_loop
    base_rho = 0.1
    current_rho = base_rho

    history = {
        "epoch_loss": [],
        "epoch_curvature": [],
        "epoch_gradnorm": [],
        "epoch_rho": [],
    }

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_gradnorm = 0.0
        n_steps = 0

        for i, batch in enumerate(batches):
            rng, k_step = random.split(rng)
            if use_sam:
                params, opt_state, loss, grad_norm = train_step_sam(
                    params, opt_state, batch, current_rho, k_step
                )
            else:
                params, opt_state, loss = train_step_standard(
                    params, opt_state, batch, k_step
                )
                grad_norm = 0.0

            total_loss += float(loss)
            total_gradnorm += float(grad_norm)
            n_steps += 1

            if i % 20 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss={float(loss):.6f}, GradNorm={float(grad_norm):.6f}, rho={current_rho:.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        avg_gradnorm = total_gradnorm / max(n_steps, 1)

        rng, key_curv = random.split(rng)
        curv = curvature_signal(params, batches[0], key_curv, long_rollout_cfg)

        current_rho = rho_schedule(base_rho, curv, ref=0.01, max_scale=5.0)

        print(f"[Epoch {epoch}] AvgLoss={avg_loss:.6f}, AvgGradNorm={avg_gradnorm:.6f}, Curvature≈{float(curv):.6f}, rho→{current_rho:.4f}")

        history["epoch_loss"].append(avg_loss)
        history["epoch_curvature"].append(float(curv))
        history["epoch_gradnorm"].append(avg_gradnorm)
        history["epoch_rho"].append(current_rho)

    return params, history

# ================== EVALUATION: LONG ROLLOUT & ENERGY DRIFT ==================

def eval_long_rollout(params, key, horizon=200, n_trajectories=8):
    key_init, key_model = random.split(key)
    x0 = random.uniform(key_init, (n_trajectories, 2), minval=-2.0, maxval=2.0)

    def step_true(carry, t):
        x = carry
        x_next = hamiltonian_dynamics(x)
        return x_next, x_next

    def step_model(carry, t):
        x = carry
        x_next = hnn_step(params, x)
        return x_next, x_next

    _, xs_true = lax.scan(step_true, x0, jnp.arange(horizon))
    _, xs_model = lax.scan(step_model, x0, jnp.arange(horizon))

    err = jnp.mean((xs_model - xs_true) ** 2, axis=-1)
    err_mean = jnp.mean(err, axis=1)

    q_true, p_true = xs_true[..., 0], xs_true[..., 1]
    q_model, p_model = xs_model[..., 0], xs_model[..., 1]

    E_true = true_hamiltonian(q_true, p_true)
    E_model = true_hamiltonian(q_model, p_model)

    E_true_mean = jnp.mean(E_true, axis=1)
    E_model_mean = jnp.mean(E_model, axis=1)

    return {
        "xs_true": xs_true,
        "xs_model": xs_model,
        "mse_per_t": err_mean,
        "E_true_mean": E_true_mean,
        "E_model_mean": E_model_mean,
    }

# ================== RUN TRAINING & EVAL ==================

if __name__ == "__main__":
    key = random.PRNGKey(0)
    final_params, history = train(
        key,
        epochs=5,
        batch_size=128,
        hidden_dim=512,
        lora_rank=32,
        base_lr=1e-3,
        use_sam=True,
    )
    print("History:", history)

    key_eval = random.PRNGKey(42)
    eval_res = eval_long_rollout(final_params, key_eval, horizon=200, n_trajectories=8)
    print("Final horizon MSE:", float(eval_res["mse_per_t"][-1]))
    print("Energy drift (true vs model, last step):",
          float(eval_res["E_true_mean"][-1]),
          float(eval_res["E_model_mean"][-1]))
