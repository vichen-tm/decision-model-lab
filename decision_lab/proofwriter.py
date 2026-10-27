"""Frozen ProofWriter policy application; no fitting on final rows."""
import numpy as np
from .calibration import features
from .policies import vote, NAMES

def predict(x, c):
    a = np.column_stack([np.ones(len(x)), (x - np.array(c['mean'])) / np.array(c['scale'])])
    return 1 / (1 + np.exp(-np.clip(a @ np.array(c['beta']), -30, 30)))

def apply(pred, conf, frozen):
    q = {k: np.column_stack([predict(features(conf[:, m], pred[:, m], k == 'cat'), cs[name]) for m, name in enumerate(NAMES)]) for k, cs in frozen['calibrators'].items()}
    out = {name: (pred[:, m], np.full(len(pred), 1 << m, dtype=np.uint8)) for m, name in enumerate(NAMES)}
    for z in frozen['policies']:
        f, p = (z['family'], z['params'])
        kind = f.rsplit('_', 1)[1]
        s = q[kind]
        n = len(pred)
        if f.startswith('conditional_weighted') or f.startswith('conditional_majority'):
            primary = NAMES.index(p['primary'])
            high = s[:, primary] >= p['threshold']
            w, strength = vote(pred, s if 'weighted' in f else None)
            decision = np.where(high, pred[:, primary], np.where(strength >= p['vote_threshold'], w, -1))
            calls = np.where(high, 1 << primary, 7).astype(np.uint8)
        elif f == 'cascade2_cat':
            decision = np.full(n, -1)
            calls = np.zeros(n, dtype=np.uint8)
            remaining = np.ones(n, dtype=bool)
            for name, t in zip(p['order'], p['thresholds']):
                m = NAMES.index(name)
                calls[remaining] |= 1 << m
                take = remaining & (s[:, m] >= t)
                decision[take] = pred[take, m]
                remaining[take] = False
        elif f == 'conditional_veto_cat':
            t, u = p['thresholds']
            primary = NAMES.index(p['primary'])
            verifier = NAMES.index(p['verifier'])
            high = s[:, primary] >= t
            decision = np.where(high | (s[:, verifier] >= u), pred[:, primary], -1)
            calls = np.where(high, 1 << primary, 1 << primary | 1 << verifier).astype(np.uint8)
        else:
            raise ValueError(f)
        out[f + '/' + z['target']] = (decision, calls)
    return (out, q)
