"""Small, deterministic correctness calibrators; no provider calls."""
import numpy as np


def stratified_folds(labels, k, seed):
    result = np.empty(len(labels), dtype=int)
    rng = np.random.default_rng(seed)
    for label in np.unique(labels):
        ix = np.flatnonzero(labels == label)
        rng.shuffle(ix)
        result[ix] = np.arange(len(ix)) % k
    return result


def features(confidence, prediction, category=False):
    x = np.log(np.clip(confidence, 1e-4, 1-1e-4) /
               (1-np.clip(confidence, 1e-4, 1-1e-4)))
    columns = [x]
    if category:
        # The label predicted at runtime, never the gold label.
        columns += [(prediction == c).astype(float) for c in (1, 2)]
        columns += [x * (prediction == c) for c in (1, 2)]
    return np.column_stack(columns)


def fit_predict(x_train, y_train, x_test):
    """L2 logistic correctness mapping, fixed penalty 2, no tuning."""
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale < 1e-8] = 1
    x = np.column_stack([np.ones(len(x_train)), (x_train-mean)/scale])
    xt = np.column_stack([np.ones(len(x_test)), (x_test-mean)/scale])
    beta = np.zeros(x.shape[1])
    base = np.clip(y_train.mean(), .001, .999)
    beta[0] = np.log(base/(1-base))
    penalty = np.eye(x.shape[1]) * 2
    penalty[0, 0] = .001
    for _ in range(80):
        p = 1/(1+np.exp(-np.clip(x@beta, -30, 30)))
        gradient = x.T@(p-y_train) + penalty@beta
        hessian = x.T@((p*(1-p))[:, None]*x) + penalty
        step = np.linalg.solve(hessian, gradient)
        # Damping prevents separation from creating unstable Newton steps.
        norm = np.linalg.norm(step)
        if norm > 2:
            step *= 2/norm
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return 1/(1+np.exp(-np.clip(xt@beta, -30, 30)))


def crossfit(conf, pred, correct, gold, train, test, seed):
    """Return inner OOF train and outer held-out test correctness estimates."""
    folds = stratified_folds(gold[train], 3, seed)
    outputs = {}
    for name, category in [('cal', False), ('cat', True)]:
        train_q, test_q = np.zeros((len(train), 3)), np.zeros((len(test), 3))
        for m in range(3):
            x = features(conf[:, m], pred[:, m], category)
            for f in range(3):
                inner_train, inner_test = train[folds != f], train[folds == f]
                train_q[folds == f, m] = fit_predict(x[inner_train], correct[inner_train, m], x[inner_test])
            test_q[:, m] = fit_predict(x[train], correct[train, m], x[test])
        outputs[name] = (train_q, test_q)
    return outputs
