import numpy as np


def wrap_phase(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def make_synthetic(shape=(256, 320), noise=0.05, seed=7):
    """Smooth ramp + peaks, wrapped observation, and reliability weights."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(shape[0]), -1:1:complex(shape[1])]
    truth = 10*x + 6*y + 7*np.exp(-12*((x-.25)**2+(y+.2)**2)) - 5*np.exp(-18*((x+.4)**2+(y-.35)**2))
    sigma = noise * (1 + 4*np.exp(-20*(x*x+y*y)))
    observed = wrap_phase(truth + rng.normal(scale=sigma))
    weight = 1 / (1 + sigma / max(noise, 1e-12))
    return truth, observed, weight


def rmse_aligned(estimate, truth):
    delta = np.asarray(estimate) - np.asarray(truth)
    delta -= delta.mean()
    return float(np.sqrt(np.mean(delta * delta)))
