import random
import copy

C = 0.3
T = 10

ROUND = 50

CLIENT_CNT = 100
SAMPLE_CNTS = [(i + 1) * 100 for i in range(CLIENT_CNT)]

# chosen_cli_idxs = set()

def choose_one(n, weights=[]):
    if len(weights) == 0:
        # return random.choice([])
        return random.randint(0, n - 1)
    else:
        assert(len(weights) == n)
        t = [i for i in range(n)]
        return random.choices(t, weights=weights)[0]

def choose_client(client_cnt, ratio, _weights = []):
    assert( 0 <= ratio and ratio <= 1)
    use_weights = len(_weights) > 0
    weights = copy.deepcopy(_weights)
    n = max(1, int(client_cnt * ratio))
    res = []
    clis = [i for i in range(client_cnt)]
    for i in range(n):
        j = choose_one(len(clis), weights)
        res.append(clis[j])
        # clis.remove(clis[j])
        del clis[j]
        if use_weights:
            del weights[j]
    return res


def emu_local_train_default(cli_idxs, sampe_cnts, max_round_time):
    used_time = 1.0
    used_samples = sum([sampe_cnts[i] for i in cli_idxs])
    good_cli_idxs = copy.deepcopy(cli_idxs)
    bad_cli_idxs = []
    return used_time, used_samples, good_cli_idxs, bad_cli_idxs


def emu_local_train(cli_idxs, sampe_cnts, max_round_time):
    # return emu_local_train_default(cli_idxs, sampe_cnts, max_round_time)
    BAD_CLIS = [i for i in range(90, 100)]
    used_time = 1.0
    used_samples = 0
    good_cli_idxs = []
    bad_cli_idxs = []
    for idx in cli_idxs:
        if idx in BAD_CLIS:
            # print(f"bad cli: {idx}")
            bad_cli_idxs.append(idx)
            used_time = max(used_time, max_round_time + 0.1)
        else:
            good_cli_idxs.append(idx)
            # used_time = max(used_time, max_round_time)
            used_samples += sampe_cnts[idx]
    return used_time, used_samples, good_cli_idxs, bad_cli_idxs



def emu_base(round, max_round_time, client_cnt, sample_cnts, ratio):
    total_samples = 0
    total_time = 0.0

    for r in range(round):
        cli_idxs = choose_client(client_cnt, ratio)
        lt_elasped, used_samples, _, _ = emu_local_train(cli_idxs, sample_cnts, max_round_time)
        if lt_elasped > max_round_time:
            total_time += max_round_time
            continue
        total_time += lt_elasped
        total_samples += used_samples
    
    return total_time, total_samples

def emu_advanced(round, max_round_time, client_cnt, sample_cnts, ratio):
    total_samples = 0
    total_time = 0.0
    cli_weights = [1.0 for _i in range(client_cnt)]

    for r in range(round):
        assert(len(cli_weights) == client_cnt)
        cli_idxs = choose_client(client_cnt, ratio, cli_weights)
        round_sample_cnt = sum([sample_cnts[i] for i in cli_idxs])
        lt_elasped, used_samples, good_cli_idxs, bad_cli_idxs = emu_local_train(cli_idxs, sample_cnts, max_round_time)
        # print(good_cli_idxs)
        for i in good_cli_idxs:
            # print()
            # print(i, len(cli_weights), len(sample_cnts))
            cli_weights[i] += sample_cnts[i] / round_sample_cnt
            total_samples += sample_cnts[i]
        for i in bad_cli_idxs:
            cli_weights[i] *= 0.5
            # print(f"bad cli: {i} : {cli_weights[i]}")
            # cli_weights[i] -= sample_cnts[i] / round_sample_cnt
        # print(f"bad clis: {i}")
        if lt_elasped > max_round_time:
            total_time += max_round_time
        else:
            total_time += lt_elasped
            # total_samples += used_samples
    return total_time, total_samples, cli_weights

base_t = []
base_s = []
adv_t = []
adv_s = []
adv_weights = []

N = 100

for i in range(N):
    t, s = emu_base(ROUND, T, CLIENT_CNT, SAMPLE_CNTS, C)
    base_t.append(t)
    base_s.append(s)

for i in range(N):
    t, s, w = emu_advanced(ROUND, T, CLIENT_CNT, SAMPLE_CNTS, C)
    adv_t.append(t)
    adv_s.append(s)
    adv_weights.append(w)

import numpy as np

print(f"[base time] avg: {np.mean(base_t)}, std: {np.std(base_t)}")
print(f"[base samples] avg: {np.mean(base_s)}, std: {np.std(base_s)}")
print(f"[advanced time] avg: {np.mean(adv_t)}, std: {np.std(adv_t)}")
print(f"[advanced samples] avg: {np.mean(adv_s)}, std: {np.std(adv_s)}")

# t, s, w = emu_advanced(ROUND, T, CLIENT_CNT, SAMPLE_CNTS, C)

# t, s = emu_base(ROUND, T, CLIENT_CNT, SAMPLE_CNTS, C)
