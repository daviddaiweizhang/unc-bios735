# Markov Chain Monte Carlo

```python
%pip install numpy matplotlib scipy
```

## Motivation

In the numerical integration lecture, we developed several methods for computing intractable integrals: deterministic quadrature, Monte Carlo integration, importance sampling, and the Laplace approximation. Each has limitations. Quadrature rules suffer from the curse of dimensionality. Simple Monte Carlo requires sampling directly from the target distribution. Importance sampling requires a good proposal distribution, and the variance of the estimator explodes when the proposal has lighter tails than the target. The Laplace approximation fails for multimodal or highly skewed posteriors.

Consider a Bayesian inference problem where we want to sample from a posterior distribution:

$$p(\boldsymbol{\theta} | \mathbf{y}) = \frac{p(\mathbf{y} | \boldsymbol{\theta}) \, p(\boldsymbol{\theta})}{p(\mathbf{y})} \propto p(\mathbf{y} | \boldsymbol{\theta}) \, p(\boldsymbol{\theta})$$

The normalizing constant $p(\mathbf{y}) = \int p(\mathbf{y} | \boldsymbol{\theta}) p(\boldsymbol{\theta}) \, d\boldsymbol{\theta}$ is typically intractable. We can evaluate $p(\mathbf{y} | \boldsymbol{\theta}) p(\boldsymbol{\theta})$ pointwise, but we cannot sample from $p(\boldsymbol{\theta} | \mathbf{y})$ directly. Importance sampling could work in principle, but finding a proposal that matches the posterior well in high dimensions is difficult.

**Markov chain Monte Carlo (MCMC)** solves this problem by constructing a Markov chain whose stationary distribution is the target distribution $p(\boldsymbol{\theta} | \mathbf{y})$. After running the chain long enough, the samples approximate draws from the target, and we can use them to compute posterior expectations, credible intervals, and other summaries. Unlike importance sampling, MCMC requires only the ability to evaluate the target density up to a normalizing constant.

### Running Example: Bayesian Logistic Regression

Throughout this lecture, we will use Bayesian logistic regression as a running example. We observe binary outcomes $y_i \in \{0, 1\}$ for $i = 1, \ldots, n$ with covariates $\mathbf{x}_i \in \mathbb{R}^p$:

$$y_i | \boldsymbol{\beta} \sim \text{Bernoulli}(\sigma(\mathbf{x}_i^T \boldsymbol{\beta})), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

With a normal prior $\boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2 \mathbf{I})$, the posterior is:

$$p(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) \propto \prod_{i=1}^n \sigma(\mathbf{x}_i^T \boldsymbol{\beta})^{y_i} (1 - \sigma(\mathbf{x}_i^T \boldsymbol{\beta}))^{1 - y_i} \cdot \prod_{j=1}^p \frac{1}{\sqrt{2\pi}\tau} e^{-\beta_j^2 / (2\tau^2)}$$

This posterior has no closed form. The logistic likelihood is not conjugate to the normal prior, so we cannot compute the normalizing constant analytically. MCMC provides a way to sample from this posterior directly.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

np.random.seed(42)

# Simulate logistic regression data
n = 200
p = 2
X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
beta_true = np.array([0.5, 1.5])
prob = 1 / (1 + np.exp(-X @ beta_true))
y = np.random.binomial(1, prob)

print(f"Simulated data: n={n}, p={p}")
print(f"True coefficients: beta = {beta_true}")
print(f"Observed proportion of y=1: {np.mean(y):.3f}")
```

We define the log-posterior (up to a constant), which is all MCMC needs:

```python
def log_posterior(beta, X, y, tau=10.0):
    """Log-posterior for Bayesian logistic regression (up to a constant)."""
    eta = X @ beta
    # Log-likelihood (numerically stable: log(1+exp(eta)) via logaddexp)
    log_lik = np.sum(y * eta - np.logaddexp(0, eta))
    # Log-prior: beta ~ N(0, tau^2 * I)
    log_prior = -0.5 * np.sum(beta**2) / tau**2
    return log_lik + log_prior


# Find the posterior mode (MAP estimate) for reference
res = minimize(lambda b: -log_posterior(b, X, y), x0=np.zeros(p), method="BFGS")
beta_map = res.x
print(f"MAP estimate: beta = {beta_map}")
```

### Question

Why can't we use the Laplace approximation to fully characterize this posterior? In what sense does Laplace provide only a partial answer, and what additional information does MCMC give us?

### Answer

The Laplace approximation provides a Gaussian approximation centered at the posterior mode, which gives us point estimates and approximate standard errors. However, it assumes the posterior is well-approximated by a Gaussian, which may not hold if the posterior is skewed, heavy-tailed, or multimodal. MCMC gives us the full posterior distribution as a collection of samples. From these samples, we can compute any functional of the posterior: means, medians, quantiles, credible intervals, posterior predictive distributions, and probabilities of arbitrary events (like $P(\beta_1 > 0 | \mathbf{y})$). The Laplace approximation gives a single Gaussian summary, while MCMC gives us as much detail about the posterior shape as we want.


## Markov Chains

Before introducing MCMC algorithms, we need to understand the mathematical object they construct: a Markov chain.

### Definition

A **Markov chain** is a sequence of random variables $X_0, X_1, X_2, \ldots$ where the distribution of $X_{t+1}$ depends only on $X_t$ and not on the earlier history $X_0, \ldots, X_{t-1}$. This is called the **Markov property** (or memoryless property):

$$P(X_{t+1} = x | X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} = x | X_t)$$

In the discrete case, the Markov chain is characterized by a **transition matrix** $\mathbf{P}$ where $P_{ij} = P(X_{t+1} = j | X_t = i)$. Each row of $\mathbf{P}$ sums to 1.

### A Simple Example

Consider a Markov chain with three states $\{1, 2, 3\}$ and transition matrix:

$$\mathbf{P} = \begin{pmatrix} 0.1 & 0.6 & 0.3 \\ 0.4 & 0.2 & 0.4 \\ 0.3 & 0.3 & 0.4 \end{pmatrix}$$

Starting from state 1, let us simulate the chain and track how often it visits each state:

```python
# Transition matrix
P = np.array([
    [0.1, 0.6, 0.3],
    [0.4, 0.2, 0.4],
    [0.3, 0.3, 0.4],
])

# Simulate Markov chain
n_steps = 10000
states = np.zeros(n_steps, dtype=int)
states[0] = 0  # Start in state 1 (index 0)

np.random.seed(42)
for t in range(1, n_steps):
    states[t] = np.random.choice(3, p=P[states[t - 1]])

# Empirical visit frequencies
visit_freq = np.bincount(states, minlength=3) / n_steps
print(f"Visit frequencies: {visit_freq}")
```

### Stationary Distribution

A probability distribution $\boldsymbol{\pi}$ is a **stationary distribution** of a Markov chain with transition matrix $\mathbf{P}$ if:

$$\boldsymbol{\pi}^T \mathbf{P} = \boldsymbol{\pi}^T$$

In other words, if the chain is in state distribution $\boldsymbol{\pi}$ at time $t$, it will still be in distribution $\boldsymbol{\pi}$ at time $t + 1$. Under additional conditions (irreducibility and aperiodicity, discussed below), the stationary distribution is the chain's long-run equilibrium: the fraction of time it spends in each state converges to $\boldsymbol{\pi}$ regardless of the starting state.

We can find the stationary distribution of our example chain by solving the eigenvector equation:

```python
# Find stationary distribution as left eigenvector
eigenvalues, eigenvectors = np.linalg.eig(P.T)
# The stationary distribution corresponds to eigenvalue 1
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi_stationary = np.real(eigenvectors[:, idx])
pi_stationary = pi_stationary / pi_stationary.sum()  # Normalize

print(f"Stationary distribution: {pi_stationary}")
print(f"Empirical frequencies:   {visit_freq}")
print(f"Check: pi @ P = {pi_stationary @ P}")
```

The empirical visit frequencies match the stationary distribution, confirming that the chain has converged to its equilibrium.

### Convergence to Stationarity

Not all Markov chains converge to a unique stationary distribution. Two properties guarantee convergence:

**Irreducibility:** Every state can be reached from every other state. The chain does not get "trapped" in a subset of states.

**Aperiodicity:** The chain does not cycle through states in a fixed pattern. If the chain returns to a state only at regular intervals (e.g., every 2 steps), it is periodic.

A Markov chain that is both irreducible and aperiodic is called **ergodic**. An ergodic chain has a unique stationary distribution, and the time-average of any function $g$ converges to its expectation under $\boldsymbol{\pi}$:

$$\frac{1}{T} \sum_{t=1}^T g(X_t) \xrightarrow{a.s.} E_\pi[g(X)] = \sum_x g(x) \pi(x)$$

This is the **ergodic theorem**, the Markov chain analog of the law of large numbers. It is the foundation of MCMC: if we can construct a Markov chain whose stationary distribution is our target posterior, then time-averages of the chain's samples give us posterior expectations.

```python
# Visualize convergence from different starting states
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for start_state in range(3):
    states_sim = np.zeros(5000, dtype=int)
    states_sim[0] = start_state
    np.random.seed(start_state)
    for t in range(1, 5000):
        states_sim[t] = np.random.choice(3, p=P[states_sim[t - 1]])

    # Cumulative fraction of time in state 0
    cum_frac = np.cumsum(states_sim == 0) / np.arange(1, 5001)
    axes[0].plot(cum_frac, label=f"Start state {start_state + 1}", alpha=0.8)

axes[0].axhline(pi_stationary[0], color="black", linestyle="--",
                label=f"$\\pi_1$ = {pi_stationary[0]:.3f}")
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Fraction of time in state 1")
axes[0].set_title("Convergence to Stationary Distribution")
axes[0].legend()

# Distribution over states at time t, starting from state 1
dist_t = np.zeros((100, 3))
dist_t[0] = [1, 0, 0]  # Start in state 1
for t in range(1, 100):
    dist_t[t] = dist_t[t - 1] @ P

for s in range(3):
    axes[1].plot(dist_t[:, s], label=f"$P(X_t = {s+1})$")
axes[1].set_xlabel("Step $t$")
axes[1].set_ylabel("Probability")
axes[1].set_title("Distribution Over States (Start from State 1)")
axes[1].legend()

plt.tight_layout()
```

### Detailed Balance

A key condition for constructing MCMC algorithms is **detailed balance** (also called reversibility). A Markov chain satisfies detailed balance with respect to distribution $\boldsymbol{\pi}$ if:

$$\pi(x) \, P(x \to y) = \pi(y) \, P(y \to x) \quad \text{for all } x, y$$

This says that the "flow" of probability from $x$ to $y$ equals the flow from $y$ to $x$ under the stationary distribution. Detailed balance is a sufficient (but not necessary) condition for $\boldsymbol{\pi}$ to be the stationary distribution. To verify: summing both sides over $x$ gives $\sum_x \pi(x) P(x \to y) = \pi(y) \sum_x P(y \to x) = \pi(y)$, which is exactly the stationarity condition $\boldsymbol{\pi}^T \mathbf{P} = \boldsymbol{\pi}^T$.

MCMC algorithms construct transition kernels that satisfy detailed balance with respect to the target distribution. This ensures that the chain's stationary distribution is exactly the distribution we want to sample from.

### Question

Consider a Markov chain on states $\{A, B, C\}$ with the following transition matrix:

$$\mathbf{P} = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}$$

(a) Is this chain irreducible?

(b) Is this chain aperiodic? What is its period?

(c) Does this chain have a stationary distribution? Will time averages converge to expectations under it?

### Answer

(a) Yes, the chain is irreducible. From any state, you can reach any other state: $A \to B \to C \to A$. All states communicate.

(b) The chain is *not* aperiodic. It is periodic with period 3: starting from state $A$, the chain returns to $A$ only at steps 3, 6, 9, etc. The chain cycles deterministically through $A \to B \to C \to A \to \cdots$.

(c) The chain has a unique stationary distribution $\boldsymbol{\pi} = (1/3, 1/3, 1/3)$ (you can verify $\boldsymbol{\pi}^T \mathbf{P} = \boldsymbol{\pi}^T$). However, because the chain is periodic, the distribution at time $t$ does not converge to $\boldsymbol{\pi}$. It cycles: $P(X_t = A) = 1$ when $t$ is a multiple of 3 and 0 otherwise. Despite this, time averages do converge: over many steps, the chain spends equal time in each state, so $\frac{1}{T}\sum_{t=1}^T g(X_t) \to \frac{1}{3}[g(A) + g(B) + g(C)]$. The ergodic theorem still holds for irreducible positive-recurrent chains, even periodic ones.


## The Metropolis-Hastings Algorithm

The Metropolis-Hastings (MH) algorithm is the most general MCMC method. It constructs a Markov chain with a specified target distribution $\pi(x)$ by proposing moves from a **proposal distribution** and accepting or rejecting them with a carefully chosen probability.

### Algorithm

Given the current state $x_t$, the MH algorithm generates the next state as follows:

1. **Propose:** Draw a candidate $x^*$ from a proposal distribution $q(x^* | x_t)$.
2. **Compute the acceptance ratio:**
$$\alpha(x_t, x^*) = \min\left(1, \frac{\pi(x^*) \, q(x_t | x^*)}{\pi(x_t) \, q(x^* | x_t)}\right)$$
3. **Accept or reject:** With probability $\alpha$, set $x_{t+1} = x^*$. Otherwise, set $x_{t+1} = x_t$.

The ratio $\pi(x^*)/\pi(x_t)$ compares how "good" the proposed state is relative to the current state. The ratio $q(x_t | x^*)/q(x^* | x_t)$ corrects for any asymmetry in the proposal. Notice that $\pi$ appears only as a ratio, so the normalizing constant cancels. This is why MCMC works with unnormalized densities.

### Why Does It Work?

The acceptance probability is designed so that the resulting Markov chain satisfies detailed balance with respect to $\pi$. To verify, consider two states $x$ and $y$ and assume $\pi(y) q(x | y) \leq \pi(x) q(y | x)$ (the other case is symmetric). Then $\alpha(x, y) = \frac{\pi(y) q(x | y)}{\pi(x) q(y | x)}$ and $\alpha(y, x) = 1$. The detailed balance condition requires:

$$\pi(x) \, q(y | x) \, \alpha(x, y) = \pi(y) \, q(x | y) \, \alpha(y, x)$$

Substituting: $\pi(x) \, q(y | x) \cdot \frac{\pi(y) q(x | y)}{\pi(x) q(y | x)} = \pi(y) \, q(x | y) \cdot 1$. Both sides equal $\pi(y) q(x | y)$. So detailed balance holds, and $\pi$ is the stationary distribution of the chain.

### Random Walk Metropolis-Hastings

The most common choice of proposal is a **symmetric random walk**: $x^* = x_t + \epsilon$ where $\epsilon \sim N(0, \sigma^2 \mathbf{I})$. Since $q(x^* | x_t) = q(x_t | x^*)$ (the normal density is symmetric), the proposal ratio cancels and the acceptance ratio simplifies to:

$$\alpha(x_t, x^*) = \min\left(1, \frac{\pi(x^*)}{\pi(x_t)}\right)$$

This is the original **Metropolis algorithm** (1953). The chain always accepts moves to higher-density regions and sometimes accepts moves to lower-density regions, with the acceptance probability proportional to the density ratio.

### Implementation for Bayesian Logistic Regression

```python
def metropolis_hastings(log_target, x0, proposal_sd, n_iter, rng=None):
    """Random walk Metropolis-Hastings sampler.

    Parameters
    ----------
    log_target : callable
        Log of the target density (up to a constant).
    x0 : array of shape (d,)
        Initial state.
    proposal_sd : float or array of shape (d,)
        Standard deviation of the Gaussian proposal.
    n_iter : int
        Number of iterations.
    rng : numpy random Generator, optional
        Random number generator.

    Returns
    -------
    dict with keys: samples, log_target_values, acceptance_rate
    """
    if rng is None:
        rng = np.random.default_rng()
    d = len(x0)
    samples = np.zeros((n_iter, d))
    log_target_values = np.zeros(n_iter)
    n_accept = 0

    x_current = x0.copy()
    log_pi_current = log_target(x_current)

    for t in range(n_iter):
        # Propose
        x_proposed = x_current + rng.normal(0, proposal_sd, size=d)

        # Compute acceptance ratio (on log scale for stability)
        log_pi_proposed = log_target(x_proposed)
        log_alpha = log_pi_proposed - log_pi_current

        # Accept or reject
        if np.log(rng.uniform()) < log_alpha:
            x_current = x_proposed
            log_pi_current = log_pi_proposed
            n_accept += 1

        samples[t] = x_current
        log_target_values[t] = log_pi_current

    return {
        "samples": samples,
        "log_target_values": log_target_values,
        "acceptance_rate": n_accept / n_iter,
    }
```

Let us run the sampler on our logistic regression example:

```python
rng = np.random.default_rng(42)
result_mh = metropolis_hastings(
    log_target=lambda b: log_posterior(b, X, y),
    x0=np.zeros(p),
    proposal_sd=0.15,
    n_iter=20000,
    rng=rng,
)

print(f"Acceptance rate: {result_mh['acceptance_rate']:.3f}")
print(f"Posterior mean (last 10000): {result_mh['samples'][10000:].mean(axis=0)}")
print(f"Posterior std  (last 10000): {result_mh['samples'][10000:].std(axis=0)}")
print(f"MAP estimate:                {beta_map}")
print(f"True beta:                   {beta_true}")
```

The posterior mean for $\beta_0$ (around 0.75) differs noticeably from the true value (0.5). MAP estimate from BFGS optimization gives nearly the same answer (0.74), confirming the chain has converged correctly. The difference between the estimate and the true value is due to finite-sample variability: with $n = 200$ observations, the MLE can differ from the truth. The true value falls well within the 95% credible interval, as expected.


### Visualizing the Chain

```python
samples = result_mh["samples"]
burnin = 5000

fig, axes = plt.subplots(2, 3, figsize=(14, 7))

# Trace plots
for j in range(p):
    axes[0, j].plot(samples[:, j], linewidth=0.3, color="steelblue", alpha=0.7)
    axes[0, j].axhline(beta_true[j], color="red", linestyle="--",
                       label=f"True $\\beta_{j}$")
    axes[0, j].axvline(burnin, color="gray", linestyle=":", alpha=0.5,
                       label="Burn-in")
    axes[0, j].set_xlabel("Iteration")
    axes[0, j].set_ylabel(f"$\\beta_{j}$")
    axes[0, j].set_title(f"Trace Plot: $\\beta_{j}$")
    axes[0, j].legend(fontsize=8)

# Histograms (post burn-in)
for j in range(p):
    axes[1, j].hist(samples[burnin:, j], bins=50, density=True,
                    alpha=0.7, color="steelblue", edgecolor="white")
    axes[1, j].axvline(beta_true[j], color="red", linestyle="--",
                       label=f"True $\\beta_{j}$")
    axes[1, j].axvline(beta_map[j], color="green", linestyle="--",
                       label="MAP")
    axes[1, j].set_xlabel(f"$\\beta_{j}$")
    axes[1, j].set_ylabel("Density")
    axes[1, j].set_title(f"Posterior: $\\beta_{j}$")
    axes[1, j].legend(fontsize=8)

# Log-posterior trace
axes[0, 2].plot(result_mh["log_target_values"], linewidth=0.3,
                color="steelblue", alpha=0.7)
axes[0, 2].set_xlabel("Iteration")
axes[0, 2].set_ylabel("Log-posterior")
axes[0, 2].set_title("Log-Posterior Trace")

# 2D scatter of posterior samples
axes[1, 2].scatter(samples[burnin:, 0], samples[burnin:, 1],
                   s=1, alpha=0.1, color="steelblue")
axes[1, 2].plot(*beta_true, "r*", markersize=15, label="True")
axes[1, 2].plot(*beta_map, "g^", markersize=10, label="MAP")
axes[1, 2].set_xlabel("$\\beta_0$")
axes[1, 2].set_ylabel("$\\beta_1$")
axes[1, 2].set_title("Joint Posterior Samples")
axes[1, 2].legend()

plt.tight_layout()
```

### Tuning the Proposal

The proposal standard deviation $\sigma$ controls the tradeoff between exploration and acceptance. If $\sigma$ is too small, nearly all proposals are accepted but the chain moves slowly (high autocorrelation). If $\sigma$ is too large, most proposals land in low-density regions and are rejected (the chain gets stuck). An asymptotic result by Roberts, Gelman, and Gilks (1997) shows that the optimal acceptance rate for random walk MH targeting a $d$-dimensional distribution is approximately **23%** as $d \to \infty$. For low-dimensional problems ($d = 1$ or $2$), the optimal rate is higher (around 44% for $d = 1$). In practice, acceptance rates in the range of 20-50% generally indicate reasonable tuning.

```python
# Compare different proposal scales
proposal_sds = [0.01, 0.15, 2.0]
fig, axes = plt.subplots(len(proposal_sds), 2, figsize=(12, 3 * len(proposal_sds)))

for i, sd in enumerate(proposal_sds):
    rng_i = np.random.default_rng(42)
    res = metropolis_hastings(
        log_target=lambda b: log_posterior(b, X, y),
        x0=np.zeros(p),
        proposal_sd=sd,
        n_iter=5000,
        rng=rng_i,
    )
    # Trace plot for beta_1
    axes[i, 0].plot(res["samples"][:, 1], linewidth=0.5, color="steelblue")
    axes[i, 0].axhline(beta_true[1], color="red", linestyle="--")
    axes[i, 0].set_ylabel(f"$\\beta_1$")
    axes[i, 0].set_title(f"$\\sigma$ = {sd}, acceptance = {res['acceptance_rate']:.2%}")

    # Autocorrelation
    chain = res["samples"][1000:, 1]
    max_lag = 100
    acf = np.correlate(chain - chain.mean(), chain - chain.mean(), mode="full")
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]
    axes[i, 1].bar(range(max_lag), acf[:max_lag], color="steelblue", width=1)
    axes[i, 1].set_ylabel("ACF")
    axes[i, 1].set_title(f"Autocorrelation ($\\sigma$ = {sd})")

axes[-1, 0].set_xlabel("Iteration")
axes[-1, 1].set_xlabel("Lag")
plt.tight_layout()
```

With $\sigma = 0.01$, the acceptance rate is near 100% but the chain barely moves, producing highly correlated samples. With $\sigma = 2.0$, the acceptance rate is very low and the chain gets stuck at the same value for many iterations. With $\sigma = 0.15$, the chain mixes well, exploring the posterior efficiently.

### Question

Suppose you are running Metropolis-Hastings on a 2D target distribution. Your current acceptance rate is 5%. You suspect the proposal variance is too large.

(a) What would you expect the trace plot to look like?

(b) If you halve the proposal standard deviation, what qualitative effect would you expect on the acceptance rate and the autocorrelation?

(c) A colleague suggests that a 95% acceptance rate would be ideal because "almost no samples are wasted." Why is this reasoning flawed?

### Answer

(a) The trace plot would show the chain stuck at the same value for long stretches, with occasional jumps when a proposal is accepted. It would look like a step function with long flat plateaus. The chain is "stuck" most of the time.

(b) Halving the proposal standard deviation would increase the acceptance rate because smaller proposals stay closer to the current state and are more likely to have similar density. However, each accepted step is also smaller, so the chain explores more slowly. The autocorrelation would change in two competing ways: fewer rejections (reducing autocorrelation from sticking) but smaller steps (increasing autocorrelation from slow movement). Overall, the chain should mix better, since 5% acceptance is far below the optimal range.

(c) A 95% acceptance rate means the proposals are very small relative to the posterior's scale. While few samples are "wasted" by rejection, the accepted samples are all very close to each other and highly correlated. The effective number of independent samples is low despite the high acceptance rate. The optimal acceptance rate of approximately 23% balances step size against rejection rate to maximize the effective sample size per iteration. Efficiency is about independent information per iteration, not about the fraction of proposals accepted.

### Question

In the Metropolis-Hastings acceptance ratio $\alpha = \min(1, \frac{\pi(x^*) q(x_t | x^*)}{\pi(x_t) q(x^* | x_t)})$, the normalizing constant of $\pi$ cancels. Why is this property essential for Bayesian inference? What would happen if the algorithm required the normalized density?

### Answer

In Bayesian inference, the posterior $\pi(\boldsymbol{\theta}) = p(\boldsymbol{\theta} | \mathbf{y})$ is known only up to the normalizing constant $p(\mathbf{y}) = \int p(\mathbf{y} | \boldsymbol{\theta}) p(\boldsymbol{\theta}) d\boldsymbol{\theta}$. Computing this integral is precisely the intractable problem that motivated MCMC in the first place. If the algorithm required the normalized density, we would need to solve the integration problem before we could start sampling, defeating the purpose.

Because MH uses only the ratio $\pi(x^*)/\pi(x_t)$, the normalizing constants cancel: $\frac{p(\mathbf{y}|\boldsymbol{\theta}^*)p(\boldsymbol{\theta}^*)/p(\mathbf{y})}{p(\mathbf{y}|\boldsymbol{\theta}_t)p(\boldsymbol{\theta}_t)/p(\mathbf{y})} = \frac{p(\mathbf{y}|\boldsymbol{\theta}^*)p(\boldsymbol{\theta}^*)}{p(\mathbf{y}|\boldsymbol{\theta}_t)p(\boldsymbol{\theta}_t)}$. We only need to evaluate the unnormalized posterior, which is just the product of the likelihood and prior.


## Gibbs Sampling

### Motivation

Metropolis-Hastings with a random walk proposal can struggle in high dimensions. As the dimension $d$ increases, a fixed proposal scale leads to lower acceptance rates, and finding a good proposal becomes harder. **Gibbs sampling** avoids this by breaking a multivariate sampling problem into a sequence of univariate (or lower-dimensional) conditional updates.

### Algorithm

Suppose the parameter vector has $d$ components: $\boldsymbol{\theta} = (\theta_1, \theta_2, \ldots, \theta_d)$. Gibbs sampling updates each component in turn by sampling from its **full conditional distribution**, the distribution of $\theta_j$ given all other components and the data:

$$\theta_j^{(t+1)} \sim p(\theta_j | \theta_1^{(t+1)}, \ldots, \theta_{j-1}^{(t+1)}, \theta_{j+1}^{(t)}, \ldots, \theta_d^{(t)}, \mathbf{y})$$

One complete pass through all $d$ components constitutes one iteration of the Gibbs sampler. Notice that each update uses the most recently sampled values for the components that have already been updated in the current iteration.

```
Gibbs Sampling:
1. Initialize θ^(0) = (θ_1^(0), ..., θ_d^(0))
2. For t = 0, 1, 2, ...:
   For j = 1, ..., d:
     Draw θ_j^(t+1) ~ p(θ_j | θ_{-j}^(current), y)
     where θ_{-j}^(current) uses updated values for components 1..j-1
     and old values for components j+1..d
```

### Gibbs as a Special Case of Metropolis-Hastings

Gibbs sampling is actually a special case of Metropolis-Hastings where the proposal for updating $\theta_j$ is $q(\theta_j^* | \boldsymbol{\theta}) = p(\theta_j^* | \boldsymbol{\theta}_{-j}, \mathbf{y})$, the full conditional distribution. The MH acceptance ratio for this proposal is:

$$\alpha = \frac{\pi(\theta_j^*, \boldsymbol{\theta}_{-j}) \, q(\theta_j | \boldsymbol{\theta}_{-j})}{\pi(\theta_j, \boldsymbol{\theta}_{-j}) \, q(\theta_j^* | \boldsymbol{\theta}_{-j})} = \frac{p(\theta_j^*, \boldsymbol{\theta}_{-j} | \mathbf{y}) \, p(\theta_j | \boldsymbol{\theta}_{-j}, \mathbf{y})}{p(\theta_j, \boldsymbol{\theta}_{-j} | \mathbf{y}) \, p(\theta_j^* | \boldsymbol{\theta}_{-j}, \mathbf{y})}$$

Using $p(\theta_j, \boldsymbol{\theta}_{-j} | \mathbf{y}) = p(\theta_j | \boldsymbol{\theta}_{-j}, \mathbf{y}) \, p(\boldsymbol{\theta}_{-j} | \mathbf{y})$, both the numerator and denominator equal $p(\boldsymbol{\theta}_{-j} | \mathbf{y})$, so $\alpha = 1$. Every Gibbs proposal is accepted. This is the main advantage of Gibbs sampling: no rejections, so every iteration produces a new sample.

### Example: Bivariate Normal

A simple but instructive example is sampling from a bivariate normal distribution $(\theta_1, \theta_2) \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with:

$$\boldsymbol{\mu} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}$$

The full conditionals are:

$$\theta_1 | \theta_2 \sim N(\rho \theta_2, 1 - \rho^2), \qquad \theta_2 | \theta_1 \sim N(\rho \theta_1, 1 - \rho^2)$$

```python
def gibbs_bivariate_normal(rho, n_iter, rng=None):
    """Gibbs sampler for a bivariate normal with correlation rho."""
    if rng is None:
        rng = np.random.default_rng()
    samples = np.zeros((n_iter, 2))
    theta1, theta2 = 0.0, 0.0

    cond_sd = np.sqrt(1 - rho**2)

    for t in range(n_iter):
        # Update theta1 given theta2
        theta1 = rng.normal(rho * theta2, cond_sd)
        # Update theta2 given theta1
        theta2 = rng.normal(rho * theta1, cond_sd)
        samples[t] = [theta1, theta2]

    return samples


# Run for low and high correlation
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, rho in zip(axes[:2], [0.3, 0.95]):
    rng_gibbs = np.random.default_rng(42)
    samps = gibbs_bivariate_normal(rho, 2000, rng_gibbs)

    ax.plot(samps[:200, 0], samps[:200, 1], "o-", markersize=2,
            linewidth=0.5, alpha=0.5, color="steelblue")
    ax.scatter(samps[200:, 0], samps[200:, 1], s=1, alpha=0.2,
               color="steelblue")
    ax.set_xlabel("$\\theta_1$")
    ax.set_ylabel("$\\theta_2$")
    ax.set_title(f"Gibbs Sampler ($\\rho$ = {rho})")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")

# Show autocorrelation comparison
for rho, color, label in [(0.3, "steelblue", "$\\rho=0.3$"),
                           (0.95, "tab:orange", "$\\rho=0.95$")]:
    rng_gibbs = np.random.default_rng(42)
    samps = gibbs_bivariate_normal(rho, 5000, rng_gibbs)
    chain = samps[500:, 0]
    max_lag = 50
    acf = np.correlate(chain - chain.mean(), chain - chain.mean(), mode="full")
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]
    axes[2].bar(np.arange(max_lag) + (0.4 if rho == 0.95 else 0),
                acf[:max_lag], width=0.4, alpha=0.7, color=color, label=label)

axes[2].set_xlabel("Lag")
axes[2].set_ylabel("ACF")
axes[2].set_title("Autocorrelation of $\\theta_1$")
axes[2].legend()

plt.tight_layout()
```

When the correlation is low ($\rho = 0.3$), the Gibbs sampler mixes quickly because each conditional update can make large moves. When $\rho$ is high ($\rho = 0.95$), the full conditionals are narrow and the sampler moves slowly along the ridge of the distribution, resulting in high autocorrelation. This is a well-known limitation of Gibbs sampling for highly correlated parameters.

### Gibbs Sampling for Bayesian Linear Regression

A natural application of Gibbs sampling is Bayesian linear regression, where conjugate priors lead to closed-form full conditionals. Consider:

$$\mathbf{y} | \boldsymbol{\beta}, \sigma^2 \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2 \mathbf{I}), \quad \boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2 \mathbf{I}), \quad \sigma^2 \sim \text{Inv-Gamma}(a_0, b_0)$$

The full conditionals are:

$$\boldsymbol{\beta} | \sigma^2, \mathbf{y} \sim N\left(\boldsymbol{\Sigma}_\beta \mathbf{X}^T \mathbf{y} / \sigma^2, \, \boldsymbol{\Sigma}_\beta\right), \quad \boldsymbol{\Sigma}_\beta = \left(\mathbf{X}^T\mathbf{X}/\sigma^2 + \mathbf{I}/\tau^2\right)^{-1}$$

$$\sigma^2 | \boldsymbol{\beta}, \mathbf{y} \sim \text{Inv-Gamma}\left(a_0 + n/2, \, b_0 + \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2/2\right)$$

```python
def gibbs_linear_regression(X, y, tau=10.0, a0=0.01, b0=0.01,
                            n_iter=10000, rng=None):
    """Gibbs sampler for Bayesian linear regression.

    Parameters
    ----------
    X : array of shape (n, p)
    y : array of shape (n,)
    tau : float
        Prior standard deviation for beta.
    a0, b0 : float
        Inverse-Gamma prior parameters for sigma^2.
    n_iter : int
        Number of Gibbs iterations.

    Returns
    -------
    dict with keys: beta_samples, sigma2_samples
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p_dim = X.shape
    XtX = X.T @ X
    Xty = X.T @ y

    # Initialize
    sigma2 = 1.0
    beta = np.zeros(p_dim)

    beta_samples = np.zeros((n_iter, p_dim))
    sigma2_samples = np.zeros(n_iter)

    for t in range(n_iter):
        # Sample beta | sigma2, y
        precision = XtX / sigma2 + np.eye(p_dim) / tau**2
        cov_beta = np.linalg.inv(precision)
        mean_beta = cov_beta @ (Xty / sigma2)
        beta = rng.multivariate_normal(mean_beta, cov_beta)

        # Sample sigma2 | beta, y
        resid = y - X @ beta
        a_post = a0 + n / 2
        b_post = b0 + np.sum(resid**2) / 2
        sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        beta_samples[t] = beta
        sigma2_samples[t] = sigma2

    return {"beta_samples": beta_samples, "sigma2_samples": sigma2_samples}


# Simulate linear regression data
np.random.seed(42)
n_lin = 100
X_lin = np.column_stack([np.ones(n_lin), np.random.normal(0, 1, n_lin)])
beta_true_lin = np.array([2.0, -1.5])
sigma_true_lin = 1.0
y_lin = X_lin @ beta_true_lin + np.random.normal(0, sigma_true_lin, n_lin)

# Run Gibbs sampler
rng_gibbs_lr = np.random.default_rng(42)
gibbs_result = gibbs_linear_regression(X_lin, y_lin, n_iter=10000, rng=rng_gibbs_lr)

burnin_lr = 1000
print("Posterior summaries (post burn-in):")
print(f"  beta0: mean={gibbs_result['beta_samples'][burnin_lr:, 0].mean():.3f}, "
      f"std={gibbs_result['beta_samples'][burnin_lr:, 0].std():.3f} "
      f"(true={beta_true_lin[0]})")
print(f"  beta1: mean={gibbs_result['beta_samples'][burnin_lr:, 1].mean():.3f}, "
      f"std={gibbs_result['beta_samples'][burnin_lr:, 1].std():.3f} "
      f"(true={beta_true_lin[1]})")
print(f"  sigma2: mean={gibbs_result['sigma2_samples'][burnin_lr:].mean():.3f}, "
      f"std={gibbs_result['sigma2_samples'][burnin_lr:].std():.3f} "
      f"(true={sigma_true_lin**2})")
```

```python
# Visualize Gibbs sampler results
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
param_names = ["$\\beta_0$", "$\\beta_1$", "$\\sigma^2$"]
true_vals = [beta_true_lin[0], beta_true_lin[1], sigma_true_lin**2]
chains = [gibbs_result["beta_samples"][:, 0],
          gibbs_result["beta_samples"][:, 1],
          gibbs_result["sigma2_samples"]]

for j in range(3):
    # Trace plot
    axes[0, j].plot(chains[j], linewidth=0.3, color="steelblue", alpha=0.7)
    axes[0, j].axhline(true_vals[j], color="red", linestyle="--")
    axes[0, j].axvline(burnin_lr, color="gray", linestyle=":", alpha=0.5)
    axes[0, j].set_xlabel("Iteration")
    axes[0, j].set_title(f"Trace: {param_names[j]}")

    # Histogram
    axes[1, j].hist(chains[j][burnin_lr:], bins=50, density=True,
                    alpha=0.7, color="steelblue", edgecolor="white")
    axes[1, j].axvline(true_vals[j], color="red", linestyle="--",
                       label="True")
    axes[1, j].set_xlabel(param_names[j])
    axes[1, j].set_title(f"Posterior: {param_names[j]}")
    axes[1, j].legend()

plt.tight_layout()
```

### When to Use Gibbs vs. Metropolis-Hastings

| Property | Gibbs Sampling | Metropolis-Hastings |
|----------|---------------|---------------------|
| Full conditionals needed | Yes (in closed form) | No |
| Acceptance rate | Always 100% | Must be tuned |
| Tuning required | None (or minimal) | Proposal variance |
| High correlation | Slow mixing | Can be designed to handle it |
| General applicability | Needs tractable conditionals | Works for any target |

In practice, many MCMC samplers combine both: use Gibbs updates for parameters with tractable full conditionals and Metropolis-Hastings updates (within the Gibbs sweep) for parameters without closed-form conditionals. This is sometimes called **Metropolis-within-Gibbs**.

### Question

A Bayesian hierarchical model has three parameter blocks: $(\boldsymbol{\beta}, \sigma^2, \boldsymbol{u})$ where $\boldsymbol{\beta}$ are fixed effects, $\sigma^2$ is the error variance, and $\boldsymbol{u}$ are random effects.

(a) Suppose the full conditional $p(\boldsymbol{\beta} | \sigma^2, \boldsymbol{u}, \mathbf{y})$ is a multivariate normal, $p(\sigma^2 | \boldsymbol{\beta}, \boldsymbol{u}, \mathbf{y})$ is an inverse-gamma, but $p(\boldsymbol{u} | \boldsymbol{\beta}, \sigma^2, \mathbf{y})$ has no standard form. How would you design a sampler for this model?

(b) In the Gibbs step for $\sigma^2$, why does the acceptance rate not need to be tuned?

### Answer

(a) Use Metropolis-within-Gibbs: sample $\boldsymbol{\beta}$ from its multivariate normal full conditional (Gibbs step), sample $\sigma^2$ from its inverse-gamma full conditional (Gibbs step), and use a Metropolis-Hastings step for $\boldsymbol{u}$, proposing $\boldsymbol{u}^* \sim N(\boldsymbol{u}^{(t)}, \sigma_{\text{prop}}^2 \mathbf{I})$ and accepting/rejecting with the MH ratio computed using $p(\boldsymbol{u} | \boldsymbol{\beta}, \sigma^2, \mathbf{y})$.

(b) In the Gibbs step for $\sigma^2$, we sample directly from the exact full conditional distribution. Since Gibbs sampling is a special case of MH where the proposal equals the full conditional, the acceptance ratio is always 1 (every proposal is accepted). There is no proposal variance to tune because we are not using a random walk proposal.


## Convergence Diagnostics

MCMC produces correlated samples from the target distribution, but only *after* the chain has converged to its stationary distribution. In practice, we need to assess whether the chain has run long enough and whether the samples are reliable. This section covers the standard diagnostic tools.

### Burn-in

The initial portion of the chain is influenced by the starting values and may not represent the target distribution. This transient phase is called the **burn-in** period. We discard the burn-in samples before computing posterior summaries. A common practice is to discard the first 10-50% of the chain, but the appropriate burn-in depends on the problem. Trace plots help determine when the chain has "forgotten" its initial state and settled into its stationary behavior.

### Trace Plots

A **trace plot** shows the sampled values of a parameter versus iteration number. A well-mixing chain looks like a "fuzzy caterpillar" oscillating rapidly around the posterior mean. Warning signs include:

- **Trends**: the chain is still moving toward the stationary distribution
- **Long flat stretches**: the chain is stuck (low acceptance rate or poor proposal)
- **Slow drift**: the chain explores the parameter space very slowly

```python
# Demonstrate good vs. poor mixing
fig, axes = plt.subplots(2, 2, figsize=(12, 6))

# Reasonable mixing (appropriate proposal)
rng1 = np.random.default_rng(42)
res_good = metropolis_hastings(
    lambda b: log_posterior(b, X, y), np.zeros(p), 0.15, 5000, rng1
)

# Poor mixing (proposal too large)
rng2 = np.random.default_rng(42)
res_poor = metropolis_hastings(
    lambda b: log_posterior(b, X, y), np.zeros(p), 3.0, 5000, rng2
)

axes[0, 0].plot(res_good["samples"][:, 1], linewidth=0.5, color="steelblue")
axes[0, 0].set_title(f"Reasonable mixing ($\\sigma$=0.15, accept={res_good['acceptance_rate']:.0%})")
axes[0, 0].set_ylabel("$\\beta_1$")

axes[0, 1].plot(res_poor["samples"][:, 1], linewidth=0.5, color="tab:orange")
axes[0, 1].set_title(f"Poor mixing ($\\sigma$=3.0, accept={res_poor['acceptance_rate']:.0%})")
axes[0, 1].set_ylabel("$\\beta_1$")

# Histograms
axes[1, 0].hist(res_good["samples"][1000:, 1], bins=40, density=True,
                color="steelblue", alpha=0.7, edgecolor="white")
axes[1, 0].axvline(beta_true[1], color="red", linestyle="--")
axes[1, 0].set_xlabel("$\\beta_1$")

axes[1, 1].hist(res_poor["samples"][1000:, 1], bins=40, density=True,
                color="tab:orange", alpha=0.7, edgecolor="white")
axes[1, 1].axvline(beta_true[1], color="red", linestyle="--")
axes[1, 1].set_xlabel("$\\beta_1$")

plt.tight_layout()
```

### Autocorrelation

MCMC samples are correlated because each sample depends on the previous one. The **autocorrelation function (ACF)** at lag $k$ measures this dependence:

$$\hat{\rho}(k) = \frac{\sum_{t=1}^{T-k} (\theta_t - \bar{\theta})(\theta_{t+k} - \bar{\theta})}{\sum_{t=1}^{T} (\theta_t - \bar{\theta})^2}$$

A fast-mixing chain has autocorrelations that decay quickly to zero. Slow-mixing chains have slowly decaying autocorrelations, meaning many iterations are needed to obtain effectively independent samples.

### Effective Sample Size

The **effective sample size (ESS)** estimates how many independent samples the correlated MCMC chain is worth:

$$\text{ESS} = \frac{T}{1 + 2 \sum_{k=1}^{\infty} \hat{\rho}(k)}$$

where $T$ is the total number of (post burn-in) samples. The denominator accounts for autocorrelation: if consecutive samples are highly correlated, the ESS is much smaller than $T$. For example, if the autocorrelation at lag 1 is 0.9 and decays geometrically, the ESS might be only $T/20$, meaning 20,000 MCMC samples contain as much information as 1,000 independent samples.

**Rule of thumb:** An ESS of at least 400 is generally recommended for reliable posterior means and 95% credible intervals. For tail quantiles or more precise estimates, aim for ESS $\geq$ 1,000. An ESS below 100 suggests the chain needs to run longer or that mixing should be improved (e.g., by tuning the proposal). The ESS *efficiency* (ESS/$T$) indicates how well the sampler is performing: values above 10% are good, while values below 1% indicate poor mixing.

```python
def compute_ess(chain):
    """Compute effective sample size using initial positive sequence estimator.

    Uses the Geyer (1992) initial positive sequence method: autocorrelations
    are summed in consecutive pairs, and the sum stops at the first pair
    whose sum is negative.
    """
    n = len(chain)
    chain_centered = chain - chain.mean()
    # Compute autocorrelations via FFT
    fft_result = np.fft.fft(chain_centered, n=2 * n)
    acf_full = np.real(np.fft.ifft(fft_result * np.conj(fft_result)))[:n]
    acf_full = acf_full / acf_full[0]

    # Geyer's initial positive sequence: sum pairs of consecutive autocorrelations
    tau = -1.0  # will add back acf[0] = 1 in the first pair
    for k in range(0, n - 1, 2):
        pair_sum = acf_full[k] + (acf_full[k + 1] if k + 1 < n else 0.0)
        if pair_sum < 0:
            break
        tau += 2 * pair_sum
    return n / max(tau, 1.0)


# Compare ESS for good vs. poor mixing
chain_good = res_good["samples"][1000:, 1]
chain_poor = res_poor["samples"][1000:, 1]

ess_good = compute_ess(chain_good)
ess_poor = compute_ess(chain_poor)

print(f"Reasonable mixing: {len(chain_good)} samples, ESS = {ess_good:.0f} "
      f"({ess_good/len(chain_good):.1%} efficiency)")
print(f"Poor mixing:       {len(chain_poor)} samples, ESS = {ess_poor:.0f} "
      f"({ess_poor/len(chain_poor):.1%} efficiency)")
```

### Gelman-Rubin Diagnostic ($\hat{R}$)

The **Gelman-Rubin diagnostic** (Gelman and Rubin, 1992) runs multiple chains from different starting points and compares the within-chain variance to the between-chain variance. If the chains have converged to the same stationary distribution, these should be similar.

For $m$ chains of length $n$ (after discarding burn-in), let $\bar{\theta}_{j}$ be the mean of chain $j$ and $s_j^2$ its variance. Define:

$$B = \frac{n}{m-1} \sum_{j=1}^m (\bar{\theta}_j - \bar{\theta}_{\cdot\cdot})^2 \qquad \text{(between-chain variance)}$$

$$W = \frac{1}{m} \sum_{j=1}^m s_j^2 \qquad \text{(within-chain variance)}$$

The potential scale reduction factor is:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}, \quad \hat{V} = \frac{n-1}{n} W + \frac{1}{n} B$$

When $\hat{R} \approx 1$, the chains have converged to the same distribution. Values substantially above 1 (conventionally $\hat{R} > 1.1$ or $\hat{R} > 1.01$) indicate incomplete convergence.

```python
def gelman_rubin(chains):
    """Compute the Gelman-Rubin R-hat statistic.

    Parameters
    ----------
    chains : list of arrays, each of shape (n,)
        Multiple MCMC chains for the same parameter.

    Returns
    -------
    float : R-hat statistic.
    """
    m = len(chains)
    n = len(chains[0])
    chain_means = np.array([c.mean() for c in chains])
    chain_vars = np.array([c.var(ddof=1) for c in chains])
    grand_mean = chain_means.mean()

    B = n * np.var(chain_means, ddof=1)  # Between-chain variance
    W = np.mean(chain_vars)               # Within-chain variance

    V_hat = (n - 1) / n * W + B / n
    R_hat = np.sqrt(V_hat / W)
    return R_hat


# Run 4 chains from different starting points
n_chains = 4
chains_list = []

for c in range(n_chains):
    rng_c = np.random.default_rng(c * 100 + 1)
    x0_c = rng_c.normal(0, 2, size=p)
    res_c = metropolis_hastings(
        lambda b: log_posterior(b, X, y), x0_c, 0.15, 15000, rng_c
    )
    chains_list.append(res_c["samples"])

# Compute R-hat after discarding burn-in
burnin_diag = 5000
for j in range(p):
    param_chains = [ch[burnin_diag:, j] for ch in chains_list]
    rhat = gelman_rubin(param_chains)
    print(f"beta_{j}: R-hat = {rhat:.4f}")
```

```python
# Visualize multiple chains
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors_chains = ["steelblue", "tab:orange", "tab:green", "tab:red"]

for j in range(p):
    for c in range(n_chains):
        axes[j].plot(chains_list[c][:, j], linewidth=0.3,
                     color=colors_chains[c], alpha=0.6,
                     label=f"Chain {c+1}" if j == 0 else None)
    axes[j].axhline(beta_true[j], color="black", linestyle="--", linewidth=1.5)
    axes[j].axvline(burnin_diag, color="gray", linestyle=":", alpha=0.5)
    axes[j].set_xlabel("Iteration")
    axes[j].set_ylabel(f"$\\beta_{j}$")
    axes[j].set_title(f"Multiple Chains: $\\beta_{j}$")

axes[0].legend(fontsize=8)
plt.tight_layout()
```

### Diagnostic Summary

When running MCMC in practice, use the following checklist:

1. **Run multiple chains** from dispersed starting values.
2. **Inspect trace plots** for each parameter. Look for stationarity and good mixing.
3. **Check $\hat{R}$** for all parameters. Values should be close to 1 (e.g., $< 1.01$).
4. **Compute ESS** for each parameter. Ensure enough effective samples for your inference goals (a common recommendation is ESS $\geq$ 400 for reliable posterior summaries).
5. **Examine autocorrelation plots.** Rapid decay indicates good mixing.
6. **Discard burn-in** before computing posterior summaries.

No single diagnostic catches all convergence problems. Use multiple diagnostics together. Diagnostics can indicate *non*-convergence but cannot prove convergence. They provide evidence that the chain has *not failed* to converge, not that it has converged.

### Question

You run two MCMC chains for 50,000 iterations each. Both trace plots look stationary after iteration 5,000, and $\hat{R} = 1.002$ for all parameters. Your colleague argues that convergence is confirmed and runs one more chain for 10,000 iterations to "save time."

(a) Is the colleague's conclusion about convergence justified? What could go wrong?

(b) The colleague's third chain of 10,000 iterations has an ESS of 150. Is this sufficient to report posterior means and 95% credible intervals?

### Answer

(a) Convergence diagnostics can suggest that chains have not *failed* to converge, but they cannot prove convergence. Two chains might have both converged to the same local mode of a multimodal posterior, and $\hat{R}$ would be close to 1 even though neither chain has explored the full posterior. Using more chains from more dispersed starting values reduces this risk. The colleague's conclusion is reasonable but not certain.

(b) ESS of 150 is marginal. For posterior means, it is likely adequate (the Monte Carlo standard error of the mean is proportional to $1/\sqrt{\text{ESS}}$). For 95% credible intervals, which depend on accurately estimating the 2.5th and 97.5th percentiles of the posterior, ESS of 150 gives rough estimates with noticeable Monte Carlo variability. A common recommendation is ESS of at least 400 for reliable credible intervals, and some authors recommend ESS $\geq$ 1000 for tail quantiles. The colleague should run the chain longer or improve mixing.


## Applied Example: Bayesian Poisson-Lognormal GLMM

Let us return to the Poisson-lognormal model from the numerical integration lecture. In that lecture, we computed the marginal likelihood by integrating out the random effects. Here we take the Bayesian approach: we place priors on all parameters and sample from the joint posterior using MCMC.

The model is:

$$y_i | b_i \sim \text{Poisson}(\exp(\beta_0 + b_i)), \quad b_i \sim N(0, \sigma^2), \quad \beta_0 \sim N(0, 10^2), \quad \sigma^2 \sim \text{Inv-Gamma}(2, 1)$$

The full conditionals are:

- $p(\beta_0 | \mathbf{b}, \mathbf{y}, \sigma^2)$: not a standard distribution (log-concave, so we use MH)
- $p(b_i | \beta_0, y_i, \sigma^2)$: not a standard distribution (we use MH for each $b_i$)
- $p(\sigma^2 | \boldsymbol{b})$: inverse-gamma (conjugate), so we use Gibbs

This is a Metropolis-within-Gibbs sampler:

```python
def mcmc_poisson_lognormal(y, n_iter=10000, rng=None):
    """Metropolis-within-Gibbs for the Poisson-lognormal GLMM."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(y)

    # Initialize
    beta0 = 0.0
    b = np.zeros(n)
    sigma2 = 1.0

    # Storage
    beta0_samples = np.zeros(n_iter)
    sigma2_samples = np.zeros(n_iter)
    b_samples = np.zeros((n_iter, n))

    # Proposal standard deviations (tuned by hand)
    sd_beta0 = 0.1
    sd_b = 0.3
    n_accept_beta0 = 0
    n_accept_b = 0

    for t in range(n_iter):
        # --- Update beta0 via MH ---
        beta0_prop = beta0 + rng.normal(0, sd_beta0)
        # Log-posterior for beta0 (clip linear predictor to avoid overflow)
        eta_curr = np.clip(beta0 + b, None, 500)
        eta_prop = np.clip(beta0_prop + b, None, 500)
        log_post_curr = (np.sum(y * eta_curr - np.exp(eta_curr))
                         - beta0**2 / (2 * 10**2))
        log_post_prop = (np.sum(y * eta_prop - np.exp(eta_prop))
                         - beta0_prop**2 / (2 * 10**2))
        if np.log(rng.uniform()) < log_post_prop - log_post_curr:
            beta0 = beta0_prop
            n_accept_beta0 += 1

        # --- Update each b_i via MH ---
        for i in range(n):
            b_prop = b[i] + rng.normal(0, sd_b)
            eta_i_curr = min(beta0 + b[i], 500)
            eta_i_prop = min(beta0 + b_prop, 500)
            log_curr = (y[i] * eta_i_curr - np.exp(eta_i_curr)
                        - b[i]**2 / (2 * sigma2))
            log_prop = (y[i] * eta_i_prop - np.exp(eta_i_prop)
                        - b_prop**2 / (2 * sigma2))
            if np.log(rng.uniform()) < log_prop - log_curr:
                b[i] = b_prop
                n_accept_b += 1

        # --- Update sigma2 via Gibbs (conjugate) ---
        a_post = 2 + n / 2
        b_post = 1 + np.sum(b**2) / 2
        sigma2 = 1.0 / rng.gamma(a_post, 1.0 / b_post)

        beta0_samples[t] = beta0
        sigma2_samples[t] = sigma2
        b_samples[t] = b

    return {
        "beta0": beta0_samples,
        "sigma2": sigma2_samples,
        "b": b_samples,
        "accept_rate_beta0": n_accept_beta0 / n_iter,
        "accept_rate_b": n_accept_b / (n_iter * n),
    }


# Simulate data (same as numerical integration lecture)
np.random.seed(42)
n_pois = 50
beta0_true_pois = 1.0
sigma_true_pois = 0.8

b_true_pois = np.random.normal(0, sigma_true_pois, n_pois)
y_pois = np.random.poisson(np.exp(beta0_true_pois + b_true_pois))

# Run MCMC
rng_pois = np.random.default_rng(42)
pois_result = mcmc_poisson_lognormal(y_pois, n_iter=15000, rng=rng_pois)

burnin_pois = 5000
print(f"Acceptance rates: beta0={pois_result['accept_rate_beta0']:.2%}, "
      f"b_i={pois_result['accept_rate_b']:.2%}")
print("Posterior summaries (post burn-in):")
print(f"  beta0: mean={pois_result['beta0'][burnin_pois:].mean():.3f}, "
      f"95% CI=({np.percentile(pois_result['beta0'][burnin_pois:], 2.5):.3f}, "
      f"{np.percentile(pois_result['beta0'][burnin_pois:], 97.5):.3f}) "
      f"(true={beta0_true_pois})")
print(f"  sigma2: mean={pois_result['sigma2'][burnin_pois:].mean():.3f}, "
      f"95% CI=({np.percentile(pois_result['sigma2'][burnin_pois:], 2.5):.3f}, "
      f"{np.percentile(pois_result['sigma2'][burnin_pois:], 97.5):.3f}) "
      f"(true={sigma_true_pois**2:.2f})")
```

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 7))

# Trace plots
axes[0, 0].plot(pois_result["beta0"], linewidth=0.3, color="steelblue")
axes[0, 0].axhline(beta0_true_pois, color="red", linestyle="--")
axes[0, 0].axvline(burnin_pois, color="gray", linestyle=":")
axes[0, 0].set_title("Trace: $\\beta_0$")
axes[0, 0].set_xlabel("Iteration")

axes[0, 1].plot(pois_result["sigma2"], linewidth=0.3, color="steelblue")
axes[0, 1].axhline(sigma_true_pois**2, color="red", linestyle="--")
axes[0, 1].axvline(burnin_pois, color="gray", linestyle=":")
axes[0, 1].set_title("Trace: $\\sigma^2$")
axes[0, 1].set_xlabel("Iteration")

# Posterior histograms
axes[1, 0].hist(pois_result["beta0"][burnin_pois:], bins=50, density=True,
                color="steelblue", alpha=0.7, edgecolor="white")
axes[1, 0].axvline(beta0_true_pois, color="red", linestyle="--", label="True")
axes[1, 0].set_xlabel("$\\beta_0$")
axes[1, 0].set_title("Posterior: $\\beta_0$")
axes[1, 0].legend()

axes[1, 1].hist(pois_result["sigma2"][burnin_pois:], bins=50, density=True,
                color="steelblue", alpha=0.7, edgecolor="white")
axes[1, 1].axvline(sigma_true_pois**2, color="red", linestyle="--", label="True")
axes[1, 1].set_xlabel("$\\sigma^2$")
axes[1, 1].set_title("Posterior: $\\sigma^2$")
axes[1, 1].legend()

plt.tight_layout()
```

This example demonstrates a key connection: in the numerical integration lecture, we computed the marginal likelihood $L(\beta_0, \sigma^2) = \int \prod_i f(y_i | b_i) \phi(b_i) \, db_i$ by integrating out the random effects. MCMC takes the complementary Bayesian approach: instead of integrating analytically, we sample all parameters (including the random effects $b_i$) jointly from the posterior. The marginal posteriors for $\beta_0$ and $\sigma^2$ implicitly integrate out the random effects through the sampling.

### Question

In the Poisson-lognormal MCMC sampler above, we update each $b_i$ individually. With $n = 50$ subjects, each iteration requires 50 individual MH steps plus 1 MH step for $\beta_0$ and 1 Gibbs step for $\sigma^2$.

(a) Why might updating all $b_i$ values in a single block (proposing $\mathbf{b}^*$ jointly) be problematic?

(b) Conversely, what is the disadvantage of the element-wise updates?

### Answer

(a) Proposing all 50 random effects jointly means proposing in a 50-dimensional space. With a random walk proposal $\mathbf{b}^* = \mathbf{b} + \boldsymbol{\epsilon}$, the acceptance rate drops rapidly with dimension. To maintain a reasonable acceptance rate, the proposal variance must be very small, leading to slow exploration. This is the curse of dimensionality for random walk MH.

(b) Element-wise updates can be slow when the $b_i$'s are correlated with each other (through their dependence on $\beta_0$ and $\sigma^2$) or when individual conditional posteriors are very different from the joint conditional. Each $b_i$ update is a small, 1D move, and the chain may need many iterations to traverse the full parameter space. For the Poisson-lognormal model where the $b_i$'s are conditionally independent given $\beta_0$ and $\sigma^2$, element-wise updates are reasonable. For models with spatial or temporal correlation among random effects, block updates or more advanced methods (like Hamiltonian Monte Carlo, covered in the next lecture) are preferred.


## Summary

We have developed the theoretical foundation and practical tools for MCMC sampling:

1. **Markov chains** converge to a unique stationary distribution when they are irreducible and aperiodic. The ergodic theorem guarantees that time averages of chain samples converge to expectations under the stationary distribution. Detailed balance provides a sufficient condition for ensuring the chain targets the desired distribution.

2. **Metropolis-Hastings** constructs a Markov chain with a specified target distribution by proposing moves and accepting or rejecting them. It requires only the ability to evaluate the target density up to a normalizing constant. The random walk variant uses a symmetric Gaussian proposal, and acceptance rates of 20-50% generally indicate good tuning. The proposal scale must be tuned to balance exploration with acceptance.

3. **Gibbs sampling** updates each parameter by sampling from its full conditional distribution. It is a special case of MH where every proposal is accepted. It works well when full conditionals have standard forms (e.g., normal, gamma) and avoids the need for tuning proposal distributions. However, it can mix slowly when parameters are highly correlated. Metropolis-within-Gibbs combines Gibbs updates for tractable conditionals with MH updates for intractable ones.

4. **Convergence diagnostics** help assess whether the chain has reached its stationary distribution. Trace plots provide visual assessment, autocorrelation and ESS quantify mixing efficiency, and the Gelman-Rubin $\hat{R}$ statistic compares multiple chains. No single diagnostic guarantees convergence; use multiple diagnostics together.

### Connections

- **Numerical integration:** MCMC generates samples from a distribution, which can be used to compute expectations (integrals) via the ergodic theorem. Unlike importance sampling, MCMC does not require a good proposal distribution for the entire target; it adapts locally.

- **EM algorithm:** Both EM and MCMC handle latent variables, but in different ways. EM finds the MLE by iterating between computing expected sufficient statistics (E-step) and maximizing (M-step). MCMC samples from the full posterior, giving uncertainty quantification that EM does not provide. MCEM (from the EM lecture) bridges the two: it uses MCMC within the E-step.

- **Optimization:** The MAP estimate (posterior mode) can be found by optimization. MCMC provides the full posterior, which includes the MAP as a point summary but also gives credible intervals, posterior predictive distributions, and model comparison tools.

- **Next lecture (Advanced MCMC):** Hamiltonian Monte Carlo uses gradient information to make informed proposals that explore high-dimensional posteriors more efficiently. Adaptive MCMC automatically tunes proposal distributions during the run.

### References

- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. *Journal of Chemical Physics*, 21(6), 1087-1092.
- Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97-109.
- Geman, S., & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 6(6), 721-741.
- Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457-472.
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 7(1), 110-120.
- Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer. Chapters 6-8.
- Givens, G. H., & Hoeting, J. A. (2013). *Computational Statistics* (2nd ed.). Wiley. Chapters 7-8.
