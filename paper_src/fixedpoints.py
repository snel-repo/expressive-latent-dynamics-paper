import numpy as np
import pytorch_lightning as pl
import torch
from fixed_point_finder.utils import FixedPoints


def get_sys_fps(system):
    if system.name == "Lorenz":
        # From https://web.math.ucsb.edu/~jhateley/paper/lorenz.pdf
        sigma, beta, rho = [system.params[p] for p in ["sigma", "beta", "rho"]]
        # Compute known fixed points
        sqrt_beta_rho = np.sqrt(beta * (rho - 1))
        fps = np.array(
            [
                [0, 0, 0],
                [sqrt_beta_rho, sqrt_beta_rho, rho - 1],
                [-sqrt_beta_rho, -sqrt_beta_rho, rho - 1],
            ]
        )

        def jacobian(fp):
            x, y, z = fp
            return np.array(
                [
                    [-sigma, sigma, 0],
                    [rho - z, -1, -x],
                    [y, x, -beta],
                ]
            )

        # Compute Jacobians
        jacobians = np.stack([jacobian(fp) for fp in fps])
        # Modify the Jacobians for updates and bin-sized steps
        dt = 0.015008107200628729
        jacobians = jacobians * dt + np.eye(3)
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(jacobians)
    elif system.name == "Rossler":
        a, b, c = [system.params[p] for p in ["a", "b", "c"]]
        d = np.sqrt(c ** 2 - 4 * a * b)
        fps = np.array(
            [
                [(c + d) / 2, (-c - d) / (2 * a), (c + d) / (2 * a)],
                [(c - d) / 2, (-c + d) / (2 * a), (c - d) / (2 * a)],
            ]
        )

        def jacobian(fp):
            x, y, z = fp
            return np.array(
                [
                    [0, -1, -1],
                    [1, a, 0],
                    [z, 0, x - c],
                ]
            )

        # Compute Jacobians
        jacobians = np.stack([jacobian(fp) for fp in fps])
        # Modify the Jacobians for updates and bin-sized steps
        dt = 0.16881835460662842
        jacobians = jacobians * dt + np.eye(3)
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(jacobians)
    elif system.name == "Arneodo":
        a, b, c, d = [system.params[p] for p in ["a", "b", "c", "d"]]
        fps = np.array(
            [
                [0, 0, 0],
                [np.sqrt(a / d), 0, 0],
                [-np.sqrt(a / d), 0, 0],
            ]
        )

        def jacobian(fp):
            x, y, z = fp
            return np.array(
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [-a + 3 * d * x ** 2, -b, -c],
                ]
            )

        # Compute Jacobians
        jacobians = np.stack([jacobian(fp) for fp in fps])
        # Modify the Jacobians for updates and bin-sized steps
        dt = 0.0904035028815997
        jacobians = jacobians * dt + np.eye(3)
        # Compute eigenvalues
        eigvals = np.linalg.eigvals(jacobians)

    else:
        raise NotImplementedError
    return fps, eigvals


def find_fixed_points(
    model: pl.LightningModule,
    state_trajs: np.array,
    mode: str,
    n_inits=1024,
    noise_scale=0.01,
    learning_rate=1e-2,
    tol_q=1e-12,
    tol_dq=1e-20,
    tol_unique=1e-3,
    max_iters=5000,
    random_seed=0,
    device="cpu",
):

    assert mode in {"rnn", "node"}
    model = model.to(device)
    state_trajs = state_trajs.to(device)
    # Seed PyTorch
    torch.manual_seed(random_seed)
    # Prevent gradient computation for the neural ODE
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Choose random points along the observed trajectories
    n_samples, n_steps, state_dim = state_trajs.shape
    state_pts = state_trajs.reshape(-1, state_dim)
    idx = torch.randint(n_samples * n_steps, size=(n_inits,), device=device)
    states = state_pts[idx]
    # Add Gaussian noise to the sampled points
    states = states + noise_scale * torch.randn_like(states, device=device)
    # Require gradients for the states
    states = states.detach()
    initial_states = states.cpu().numpy()
    states.requires_grad = True
    # Create the optimizer
    opt = torch.optim.Adam([states], lr=learning_rate)
    # Run the optimization
    iter_count = 1
    q_prev = torch.full((n_inits,), float("nan"), device=device)
    while True:
        # Perform forward pass
        if mode == "rnn":
            input_size = model.decoder.cell.input_size
            inputs = torch.zeros(n_inits, input_size, device=device)
            F = model.decoder.cell(inputs, states)
            q = 0.5 * torch.sum((F.squeeze() - states.squeeze()) ** 2, dim=1)
        elif mode == "node":
            q = 0.5 * torch.sum(model.decoder.vf(None, states) ** 2, dim=1)
        dq = torch.abs(q - q_prev)
        q_scalar = torch.mean(q)
        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()
        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        # Report progress
        if iter_count % 500 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(f"\nIteration {iter_count}/{max_iters}")
            print(f"q = {mean_q:.2E} +/- {std_q:.2E}")
            print(f"dq = {mean_dq:.2E} +/- {std_dq:.2E}")
        # Check termination criteria
        converged = np.all(np.logical_or(dq_np < tol_dq * learning_rate, q_np < tol_q))
        if iter_count > 1 and converged:
            print("Optimization complete to desired tolerance.")
            break
        if iter_count + 1 > max_iters:
            print("Maximum iteration count reached. Terminating.")
            break
        q_prev = q
        iter_count += 1
    # Collect fixed points
    qstar = q.cpu().detach().numpy()
    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
        tol_unique=tol_unique,
    )
    unique_fps = all_fps.get_unique()
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist(np.log10(unique_fps.qstar+1e-20), bins=50)
    # plt.savefig("hist.png")
    # plt.close()
    # import pdb; pdb.set_trace()
    # Reject FPs outside the tolerance
    best_fps = unique_fps.qstar < tol_q
    best_fps = FixedPoints(
        xstar=unique_fps.xstar[best_fps],
        x_init=unique_fps.x_init[best_fps],
        qstar=unique_fps.qstar[best_fps],
        dq=unique_fps.dq[best_fps],
        n_iters=unique_fps.n_iters[best_fps],
        tol_unique=tol_unique,
    )

    # Compute the Jacobian for each fixed point
    def J_func(x):
        x = x[None, :]
        if mode == "rnn":
            input_size = model.decoder.cell.input_size
            inputs = torch.zeros(1, input_size, device=device)
            F = model.decoder.cell(inputs, x)
        elif mode == "node":
            F = 1 / 70 * model.decoder.vf(None, x) + x
        return F.squeeze()

    all_J = []
    x = torch.tensor(best_fps.xstar, device=device)
    for i in range(best_fps.n):
        single_x = x[i, :]
        J = torch.autograd.functional.jacobian(J_func, single_x)
        all_J.append(J)
    # Recombine and decompose Jacobians for the whole batch
    dFdx = torch.stack(all_J).cpu().detach().numpy()
    best_fps.J_xstar = dFdx
    best_fps.decompose_jacobians()

    return best_fps
