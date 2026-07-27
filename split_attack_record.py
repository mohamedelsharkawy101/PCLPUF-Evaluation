"""
PC-LPUF Split Attack
====================
This script implements a divide-and-conquer split attack for the PC-LPUF.
It follows the same high-level idea as the iPUF split attack from Wisiol et al.
(CHES 2020): train a lower model from imperfectly obfuscated CRPs, recover
upper-layer labels from the lower model, train upper-layer PUF models, and
iteratively refine both parts.

During the run, the script records the predicted-response accuracy (not seen by the attacker for analysis only) of each
upper PUF across all rounds and saves the results to a CSV file for analysis.
"""

import numpy as np
import csv
import os
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import accuracy_score
from itertools import product as iterproduct


# ──────────────────────────────────────────────────────────────────────────────
#  1.  REAP-NVM PUF
# ──────────────────────────────────────────────────────────────────────────────



def ReapNVM(num_bits, seed, sigma_proc=0.05):
    rng = np.random.default_rng(seed)

    chal_length = num_bits
    n_levels = 4

    # ---- Your 5 fixed nominal resistance levels ----
    R_levels_nom = np.array([10e3,75e3,125e3 ,275e3])

    # ---- Convert to log domain ----
    log_R_levels_nom = np.log10(R_levels_nom)

    # ---- Allocate output arrays ----
    tR = np.zeros((2, n_levels, chal_length))

    # ---- Generate per-cell values ----
    for row in range(2):
        for stage in range(chal_length):

            # Add Gaussian process variation in log domain
            log_levels = log_R_levels_nom + sigma_proc * rng.standard_normal(n_levels)

            # Convert back to linear domain
            levels = 10 ** log_levels

            # RC delay mapping
            tR[row, :, stage] = levels * 250e-12

    # ---- Deterministic switching delay ----
    tSW = np.full((2, 2, chal_length), 372.0 / 1e12)

    return 4.0 * tR, 4.0 * tSW

def ReapNVM_evaluate(PUF, challenge, position, value, chunk_size=100_000):
    chal      = (challenge + 1) / 2.0
    tR4, tSW4 = PUF
    chalpos   = position.astype(int).flatten()
    chalval   = value.astype(int).flatten()
    N         = chal.shape[0]
    responses = np.zeros(N, dtype=np.int8)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        n   = end - start
        cc  = chal[start:end]
        pc  = chalpos[start:end]
        vc  = chalval[start:end]
        tv1 = np.tile(tR4[0, 0, :], (n, 1)); tv2 = np.tile(tR4[1, 0, :], (n, 1))
        tv1[np.arange(n), pc] = tR4[0, vc, pc]
        tv2[np.arange(n), pc] = tR4[1, vc, pc]
        c  = np.bitwise_xor.accumulate(cc.astype(np.uint8), axis=1)
        t1 = np.sum(np.where(c == 0, tv1 + tSW4[0, 0, :], tv2 + tSW4[1, 0, :]), axis=1)
        t2 = np.sum(np.where(c == 0, tv2 + tSW4[0, 1, :], tv1 + tSW4[1, 1, :]), axis=1)
        responses[start:end] = (t1 > t2).astype(np.int8)
    return responses


# ──────────────────────────────────────────────────────────────────────────────
#  2.  APUF
# ──────────────────────────────────────────────────────────────────────────────

def apuf_generate(k, chal_size, seed=0):
    return np.random.default_rng(seed).normal(0, 1, (k, chal_size + 1))

def apuf_response(w, Phi):
    return (Phi @ w <= 0).astype(np.int8)


# ──────────────────────────────────────────────────────────────────────────────
#  3.  XOR Obfuscation
# ──────────────────────────────────────────────────────────────────────────────

def dec_to_bin_vec(x, bitlen):
    return np.array([(x >> i) & 1 for i in range(bitlen)][::-1], dtype=np.uint8)

def bin_vec_to_dec(bits):
    out = 0
    for b in bits: out = (out << 1) | int(b)
    return out

def sliding_window_xor(bits, window_bits):
    window_bits = window_bits.astype(bits.dtype)
    for start in range(0, bits.shape[0], window_bits.shape[0]):
        end = min(start + window_bits.shape[0], bits.shape[0])
        bits[start:end] ^= window_bits[:end - start]
    return bits

def xor_obfuscate_position_value(position, value, upper_resp):
    """
    upper_resp : (K_UP, N) in {0,1}
    position   : (N,)
    value      : (N,)
    """
    N = position.shape[0]
    pos_out = np.zeros(N, dtype=np.uint32)
    val_out = np.zeros(N, dtype=np.uint32)
    for n in range(N):
        window_bits = upper_resp[:, n]
        pos_bits    = sliding_window_xor(dec_to_bin_vec(position[n], 7), window_bits)
        val_bits    = sliding_window_xor(dec_to_bin_vec(value[n],    2), window_bits)
        pos_out[n]  = bin_vec_to_dec(pos_bits)
        val_out[n]  = bin_vec_to_dec(val_bits)
    return pos_out, val_out


# ──────────────────────────────────────────────────────────────────────────────
#  4.  PC-LPUF Evaluate
# ──────────────────────────────────────────────────────────────────────────────

def pclpuf_evaluate(upper_w, lower_pufs, challenges, position, value):
    K_UP = upper_w.shape[0]
    N    = challenges.shape[0]
    Phi  = transform(challenges)
    upper_resp = np.array([apuf_response(upper_w[i], Phi) for i in range(K_UP)])
    pos_eff, val_eff = xor_obfuscate_position_value(position, value, upper_resp)
    xor_resp = np.zeros(N, dtype=np.int8)
    for puf in lower_pufs:
        xor_resp = np.bitwise_xor(xor_resp,
                   ReapNVM_evaluate(puf, challenges, pos_eff, val_eff))
    return xor_resp


def lower_layer_evaluate(lower_pufs, challenges, pos_eff, val_eff):
    N        = challenges.shape[0]
    xor_resp = np.zeros(N, dtype=np.int8)
    for puf in lower_pufs:
        xor_resp = np.bitwise_xor(xor_resp,
                   ReapNVM_evaluate(puf, challenges, pos_eff, val_eff))
    return xor_resp


# ──────────────────────────────────────────────────────────────────────────────
#  5.  Parity Transform + Feature Builder
# ──────────────────────────────────────────────────────────────────────────────

def transform(challenges):
    N = challenges.shape[0]
    return np.hstack([np.cumprod(challenges, axis=1), np.ones((N, 1))])


def prepare_lr_features(Phi, positions, values, chal_size, n_levels):
    N         = Phi.shape[0]
    pos       = positions.astype(int)
    val       = values.astype(int)
    delta_idx = pos * n_levels + val
    data      = Phi[np.arange(N), pos]
    X_delta   = sparse.csr_matrix((data, (np.arange(N), delta_idx)),
                                   shape=(N, chal_size * n_levels))
    return sparse.hstack([sparse.csr_matrix(Phi), X_delta], format='csr')


# ──────────────────────────────────────────────────────────────────────────────
#  6.  XOR LR Loss and Gradient
# ──────────────────────────────────────────────────────────────────────────────

def xor_lr_loss_and_grad(w_flat, X_list, y, K, feat_size, lam=1e-4):
    N  = X_list[0].shape[0]
    w  = w_flat.reshape(K, feat_size)
    margins   = np.array([X_list[k].dot(w[k]) for k in range(K)])
    signs     = np.sign(margins)
    log_abs   = np.log(np.abs(margins) + 1e-12)
    prod_sign = np.prod(signs, axis=0)
    log_sum   = np.sum(log_abs, axis=0)
    product   = prod_sign * np.exp(np.clip(log_sum, -500, 500))
    p         = np.clip(expit(product), 1e-12, 1 - 1e-12)
    loss      = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    loss     += lam * 0.5 * np.dot(w_flat, w_flat)
    err       = (p - y) / N
    grad      = np.zeros_like(w)
    for k in range(K):
        others  = np.prod(margins[np.arange(K) != k], axis=0)
        grad[k] = X_list[k].T.dot(err * others) + lam * w[k]
    return loss, grad.flatten()


def predict_xor_lr(w_flat, X_list, K, feat_size):
    w         = w_flat.reshape(K, feat_size)
    margins   = np.array([X_list[k].dot(w[k]) for k in range(K)])
    prod_sign = np.prod(np.sign(margins), axis=0)
    log_sum   = np.sum(np.log(np.abs(margins) + 1e-12), axis=0)
    product   = prod_sign * np.exp(np.clip(log_sum, -500, 500))
    return (product <= 0).astype(np.int8)


def train_xor_lr(X_train, y_train, K, lam=1e-4, max_iter=1000, w0=None, seed=42):
    feat_size = X_train.shape[1]
    X_sp      = sparse.csr_matrix(X_train)
    X_list    = [X_sp for _ in range(K)]
    if w0 is None:
        np.random.seed(seed)
        w0 = np.random.randn(K * feat_size) * 0.01
    result = minimize(
        fun=xor_lr_loss_and_grad, x0=w0,
        args=(X_list, y_train.astype(float), K, feat_size, lam),
        jac=True, method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
    )
    return result.x, feat_size


def eval_model(w_flat, X_sp, K, feat_size):
    return predict_xor_lr(w_flat, [X_sp for _ in range(K)], K, feat_size)


# ──────────────────────────────────────────────────────────────────────────────
#  7.  Phase 1 — Random upper bit guess
# ──────────────────────────────────────────────────────────────────────────────

def interpose_random_pclpuf(challenges, position, value,
                             K_UP, chal_size, n_levels, seed=0):
    N   = challenges.shape[0]
    rng = np.random.default_rng(seed)
    Phi = transform(challenges)
    guessed_upper = rng.integers(0, 2, size=(K_UP, N)).astype(np.int8)
    pos_chosen, val_chosen = xor_obfuscate_position_value(
        position, value, guessed_upper)
    return prepare_lr_features(Phi, pos_chosen, val_chosen, chal_size, n_levels)


# ──────────────────────────────────────────────────────────────────────────────
#  8.  Build Upper PUF Training Set
# ──────────────────────────────────────────────────────────────────────────────

def build_upper_training_set(challenges, position, value, y_pclpuf,
                              w_down, K_d, feat_size_down,
                              chal_size, n_levels, K_UP,
                              block_size=10_000):
    combos    = np.array(list(iterproduct([0, 1], repeat=K_UP)), dtype=np.int8)
    N         = challenges.shape[0]
    sel_chals = []
    sel_resps = []

    for idx in range(0, N, block_size):
        bc  = challenges[idx:idx+block_size]
        bp  = position[idx:idx+block_size]
        bv  = value[idx:idx+block_size]
        br  = y_pclpuf[idx:idx+block_size]
        bn  = len(bc)
        phi = transform(bc)

        all_preds = np.zeros((len(combos), bn), dtype=np.int8)
        for c_idx, combo in enumerate(combos):
            ur           = np.tile(combo.reshape(K_UP, 1), (1, bn))
            pos_c, val_c = xor_obfuscate_position_value(bp, bv, ur)
            X_c          = prepare_lr_features(phi, pos_c, val_c, chal_size, n_levels)
            all_preds[c_idx] = eval_model(w_down, sparse.csr_matrix(X_c),
                                          K_d, feat_size_down)

        matches = (all_preds == br[np.newaxis, :])

        n_insensitive = 0
        n_model_wrong = 0

        for n in range(bn):
            correct_idxs = np.where(matches[:, n])[0]
            if len(correct_idxs) == 0:
                n_model_wrong += 1
                continue
            if len(correct_idxs) == len(combos):
                n_insensitive += 1

            chosen_combo = combos[correct_idxs[np.random.randint(len(correct_idxs))]]
            sel_chals.append(bc[n])
            sel_resps.append(chosen_combo)

        if (idx + block_size) % 50_000 == 0:
            print(f"    {min(idx+block_size, N)}/{N} processed, "
                  f"{len(sel_chals)} selected | "
                  f"insensitive={n_insensitive} model_wrong={n_model_wrong}")

    if len(sel_chals) == 0:
        return None, None

    return np.array(sel_chals), np.array(sel_resps)


# ──────────────────────────────────────────────────────────────────────────────
#  9.  Full Split Attack
# ──────────────────────────────────────────────────────────────────────────────

def split_attack(upper_w, lower_pufs, chal_size, n_levels, K_UP, K_d,
                 n_train, n_test, max_rounds=100, target_acc=0.90,
                 lam=1e-4, max_iter=500):

    print(f"\n{'='*60}")
    print(f"  PC-LPUF Split Attack")
    print(f"  chal_size={chal_size}, K_UP={K_UP}, K_d={K_d}")
    print(f"  CRPs : train={n_train:,}  test={n_test:,}")
    print(f"  max_rounds={max_rounds}")
    print(f"{'='*60}")

    # ── Generate CRPs ─────────────────────────────────────────────────────────
    print("\n[1] Generating CRPs...")
    np.random.seed(60)
    challenges_tr = np.random.choice([-1, 1], size=(n_train, chal_size))
    pos_tr        = np.random.randint(0, chal_size, n_train)
    val_tr        = np.random.randint(0, n_levels,  n_train)
    challenges_te = np.random.choice([-1, 1], size=(n_test,  chal_size))
    pos_te        = np.random.randint(0, chal_size, n_test)
    val_te        = np.random.randint(0, n_levels,  n_test)

    y_train = pclpuf_evaluate(upper_w, lower_pufs, challenges_tr, pos_tr, val_tr)
    y_test  = pclpuf_evaluate(upper_w, lower_pufs, challenges_te, pos_te, val_te)
    print(f"    Response balance: {y_train.mean()*100:.1f}% ones")

    Phi_te       = transform(challenges_te)
    y_lower_true = lower_layer_evaluate(lower_pufs, challenges_te, pos_te, val_te)

    # ── Phase 1: Train lower model with random upper bit guesses ──────────────
    print(f"\n[2] Phase 1 — Training lower model with random upper guesses...")
    Phi_train = interpose_random_pclpuf(
        challenges_tr, pos_tr, val_tr, K_UP, chal_size, n_levels, seed=30
    )
    w_down, feat_size_down = train_xor_lr(
        Phi_train, y_train, K_d, lam=lam, max_iter=max_iter, seed=42
    )

    X_te_base = [prepare_lr_features(Phi_te, pos_te, val_te,
                                      chal_size, n_levels) for _ in range(K_d)]
    pred_lo   = predict_xor_lr(w_down, X_te_base, K_d, feat_size_down)
    acc_lo_p1 = max(accuracy_score(y_lower_true, pred_lo),
                    1 - accuracy_score(y_lower_true, pred_lo))
    print(f"    Phase 1 lower model accuracy: {acc_lo_p1*100:.2f}%")

    # ── Per-round tracking ────────────────────────────────────────────────────
    # We collect one record per round of the 50-round loop.
    # Each record stores the lower-model accuracy, the overall upper-PUF accuracy,
    # the per-upper-PUF accuracies (from the predicted responses of all upper PUFs),
    # and the final full-model accuracy for that round.
    round_records = []

    w_up         = None
    feat_size_up = None

    for rnd in range(max_rounds):
        print(f"\n[Round {rnd+1}/{max_rounds}]")

        # ── Build upper training set ──────────────────────────────────────────
        print("  Building upper training set...")
        sel_C, sel_r = build_upper_training_set(
            challenges_tr, pos_tr, val_tr, y_train,
            w_down, K_d, feat_size_down,
            chal_size, n_levels, K_UP
        )

        if sel_C is None or len(sel_C) < 50:
            print("  WARNING: Not enough challenges selected. Stopping early.")
            break

        print(f"  Selected {len(sel_C):,} / {n_train:,} challenges "
              f"({len(sel_C)/n_train*100:.1f}%)")

        # ── Train K_UP independent APUF models ────────────────────────────────
        Phi_up_sel   = transform(sel_C)
        feat_size_up = Phi_up_sel.shape[1]
        new_w_up     = []
        for i in range(K_UP):
            w0_i   = None if w_up is None else w_up[i]
            w_i, _ = train_xor_lr(
                Phi_up_sel, sel_r[:, i],
                K=1, lam=lam, max_iter=max_iter,
                w0=w0_i, seed=43 + rnd + i
            )
            new_w_up.append(w_i)
        w_up = new_w_up

        # ── Evaluate each APUF individually ───────────────────────────────────
        Phi_te_sp     = sparse.csr_matrix(Phi_te)
        upper_resp_te = np.array([apuf_response(upper_w[i], Phi_te)
                                  for i in range(K_UP)])
        acc_up_indiv  = []
        for i in range(K_UP):
            pred_i = eval_model(w_up[i], Phi_te_sp, 1, feat_size_up)
           # acc_i  = max(accuracy_score(upper_resp_te[i], pred_i),
           #              1 - accuracy_score(upper_resp_te[i], pred_i))
            
            acc_i  = accuracy_score(upper_resp_te[i], pred_i)
            
            acc_up_indiv.append(acc_i)

        acc_up_mean = float(np.mean(acc_up_indiv))
        print(f"  Upper APUF individual accs : "
              f"{[f'{a*100:.1f}%' for a in acc_up_indiv]}")
        print(f"  Upper APUF mean accuracy   : {acc_up_mean*100:.2f}%")

        # ── Retrain lower with predicted upper bits ────────────────────────────
        print("  Retraining lower model...")
        Phi_tr_sp       = sparse.csr_matrix(transform(challenges_tr))
        pred_upper_resp = np.array([
            eval_model(w_up[i], Phi_tr_sp, 1, feat_size_up)
            for i in range(K_UP)
        ])
        pos_tr_pred, val_tr_pred = xor_obfuscate_position_value(
            pos_tr, val_tr, pred_upper_resp
        )
        X_train_new = prepare_lr_features(
            transform(challenges_tr), pos_tr_pred, val_tr_pred,
            chal_size, n_levels
        )
        w_down, feat_size_down = train_xor_lr(
            X_train_new, y_train, K_d, lam=lam, max_iter=max_iter,
            w0=w_down, seed=44 + rnd
        )

        # ── Lower model accuracy ───────────────────────────────────────────────
        X_te2    = [prepare_lr_features(Phi_te, pos_te, val_te,
                                         chal_size, n_levels) for _ in range(K_d)]
        pred_lo2 = predict_xor_lr(w_down, X_te2, K_d, feat_size_down)
        acc_lo   = max(accuracy_score(y_lower_true, pred_lo2),
                       1 - accuracy_score(y_lower_true, pred_lo2))
        print(f"  Lower model accuracy       : {acc_lo*100:.2f}%")

        # ── Full PC-LPUF accuracy ──────────────────────────────────────────────
        pred_upper_te = np.array([
            eval_model(w_up[i], Phi_te_sp, 1, feat_size_up)
            for i in range(K_UP)
        ])
        pos_te_pred, val_te_pred = xor_obfuscate_position_value(
            pos_te, val_te, pred_upper_te
        )
        X_te_full = [prepare_lr_features(Phi_te, pos_te_pred, val_te_pred,
                                          chal_size, n_levels) for _ in range(K_d)]
        pred_full = predict_xor_lr(w_down, X_te_full, K_d, feat_size_down)
        acc_full  = max(accuracy_score(y_test, pred_full),
                        1 - accuracy_score(y_test, pred_full))
        print(f"  Full PC-LPUF accuracy      : {acc_full*100:.2f}%")

        # ── Store this round ───────────────────────────────────────────────────
        # This saves the predicted-response performance of every upper PUF
        # for the current round, together with the lower and full-model metrics.
        record = {
            'round'           : rnd + 1,
            'lower_acc'       : acc_lo,
            'upper_mean_acc'  : acc_up_mean,
            'upper_indiv_accs': acc_up_indiv,   # list of K_UP floats
            'full_acc'        : acc_full,
        }
        round_records.append(record)

        if acc_full >= target_acc:
            print(f"\n  ✓ Target {target_acc*100:.0f}% reached at round {rnd+1}!")
            break

    # ── Write all rounds to CSV ────────────────────────────────────────────────
    results_file = f'pclpuf_split_K_UP{K_UP}_Kd{K_d}_train{n_train}.csv'
    file_exists  = os.path.isfile(results_file)

    # Header: fixed columns + one column per upper APUF.
    # These columns record the predicted-response accuracy of each upper PUF
    # at every round of the 50-round experiment.
    indiv_headers = [f'upper_apuf_{i+1}_acc' for i in range(K_UP)]
    header = (['K_UP', 'K_d', 'chal_size', 'n_levels', 'n_train', 'n_test',
                'phase1_lower_acc', 'round', 'lower_acc',
                'upper_mean_acc'] + indiv_headers + ['full_acc'])

    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for rec in round_records:
            indiv_vals = [f'{a*100:.4f}' for a in rec['upper_indiv_accs']]
            row = ([K_UP, K_d, chal_size, n_levels, n_train, n_test,
                    f'{acc_lo_p1*100:.4f}',
                    rec['round'],
                    f'{rec["lower_acc"]*100:.4f}',
                    f'{rec["upper_mean_acc"]*100:.4f}']
                   + indiv_vals
                   + [f'{rec["full_acc"]*100:.4f}'])
            writer.writerow(row)

    print(f"\n{'─'*60}")
    print(f"  All {len(round_records)} rounds saved to: {results_file}")
    print(f"  Columns: round, lower_acc, upper_mean_acc, "
          f"upper_apuf_1..{K_UP}_acc, full_acc")
    print(f"{'─'*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
#  10.  Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    chal_size = 128
    n_levels  = 4
    K_UP      = 4
    K_d       = 3
    n_train   = 800_001
    n_test    =  40_000

    np.random.seed(60)
    upper_w    = apuf_generate(K_UP, chal_size, seed=0)
    lower_pufs = [ReapNVM(chal_size, seed=42 + k) for k in range(K_d)]

    split_attack(
        upper_w    = upper_w,
        lower_pufs = lower_pufs,
        chal_size  = chal_size,
        n_levels   = n_levels,
        K_UP       = K_UP,
        K_d        = K_d,
        n_train    = n_train,
        n_test     = n_test,
        max_rounds = 50,          # ← changed from 5 to 100
        target_acc = 0.90,
        lam        = 1e-4,
        max_iter   = 500,
    )