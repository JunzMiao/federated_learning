import os
import json
import matplotlib.pyplot as plt

# class Metric(object):
#     def __init__(self, pred_stat, label_cnt):
#        self.pred_stat = pred_stat 
#        self.label_cnt = label_cnt
#        self.calc_metrics()

def calc_metrics(pred_stat, label_cnt):
# def calc_metrics(self):
    # pred_stat = self.pred_stat
    # label_cnt = self.label_cnt

    epsilon = 1e-6
    def _satisfied_cnt(f):
        return len(list(filter(f, pred_stat)))
    # acc
    # acc = len(list(filter(lambda x : x[0] == x[1], s)))
    acc = _satisfied_cnt(lambda x : x[0] == x[1])
    acc /= _satisfied_cnt(lambda x : True)

    # precision
    prs = []
    for i in range(label_cnt):
        # v1 = len(list(filter(lambda x : x[0] == i and x[1] == i)))
        # print(i)
        p = _satisfied_cnt(lambda x : x[0] == i and x[1] == i)
        p /= _satisfied_cnt(lambda x : x[1] == i) + p + epsilon
        prs.append(p)
    
    # detection rate
    drs = []
    for i in range(label_cnt):
        dr = _satisfied_cnt(lambda x : x[0] == i and x[1] == i)
        dr /= _satisfied_cnt(lambda x : x[0] == i) + dr + epsilon
        drs.append(dr)
    
    # false positive rate
    fprs = []
    for i in range(label_cnt):
        fpr = _satisfied_cnt(lambda x : x[1] == i and x[0] != i)
        fpr /= (fpr +  _satisfied_cnt(lambda x : x[0] != i and x[1] != i)) + epsilon
        fprs.append(fpr)
    
    pr = _satisfied_cnt(lambda x : x[0] != 0 and x[1] != 0)
    pr /= _satisfied_cnt(lambda x : x[1] != 0) + epsilon
    dr = _satisfied_cnt(lambda x : x[0] != 0 and x[1] != 0)
    dr /= _satisfied_cnt(lambda x : x[0] != 0) + epsilon
    fpr = _satisfied_cnt(lambda x : x[0] == 0 and x[1] != 0)
    fpr /= _satisfied_cnt(lambda x : x[1] != 0) + epsilon
    
    # # OOP version
    # self.acc = acc
    # self.pr = pr
    # self.dr = dr
    # self.fpr = fpr
    # self.prs =prs 
    # self.drs = drs
    # self.fprs = fprs

    m = {}
    m["acc"] = acc
    m["pr"] = pr
    m["dr"] = dr
    m["fpr"] = fpr
    m["prs"] =prs 
    m["drs"] = drs
    m["fprs"] = fprs
    return m
    # return acc, prs, drs, fprs, pr, dr, fpr

accs = lambda ms : list(map(lambda m : m["acc"], ms))
prs = lambda ms : list(map(lambda m : m["pr"], ms))
drs = lambda ms : list(map(lambda m : m["dr"], ms))
fprs = lambda ms : list(map(lambda m : m["fpr"], ms))
iprs = lambda ms, i : list(map(lambda m : m["prs"][i], ms))
idrs = lambda ms, i : list(map(lambda m : m["drs"][i], ms))
ifprs = lambda ms, i : list(map(lambda m : m["fprs"][i], ms))

# # OOP version
# accs = lambda ms : list(map(lambda m : m.acc, ms))
# prs = lambda ms : list(map(lambda m : m.pr, ms))
# drs = lambda ms : list(map(lambda m : m.dr, ms))
# fprs = lambda ms : list(map(lambda m : m.fpr, ms))
# iprs = lambda ms, i : list(map(lambda m : m.prs[i], ms))
# idrs = lambda ms, i : list(map(lambda m : m.drs[i], ms))
# ifprs = lambda ms, i : list(map(lambda m : m.fprs[i], ms))

def save_res(res_dir, conf, losses, metrics, l_usrs_losses=[], l_usrs_metrics=[]):
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    open(f"{res_dir}/conf.txt", "w").write(json.dumps(conf))
    open(f"{res_dir}/losses.txt", "w").write(json.dumps(losses))
    open(f"{res_dir}/metrics.txt", "w").write(json.dumps(metrics))
    if len(l_usrs_losses) != 0 and len(l_usrs_metrics) != 0:
        open(f"{res_dir}/l_usrs_losses.txt", "w").write(json.dumps(l_usrs_losses))
        open(f"{res_dir}/l_usrs_metrics.txt", "w").write(json.dumps(l_usrs_metrics))

def load_res(res_dir, has_usr_records=False):
    conf = json.loads(open(f"{res_dir}/conf.txt", "r").read())
    losses = json.loads(open(f"{res_dir}/losses.txt", "r").read())
    metrics = json.loads(open(f"{res_dir}/metrics.txt", "r").read())
    if has_usr_records:
        l_usrs_losses = json.loads(open(f"{res_dir}/l_usrs_losses.txt", "r").read())
        l_usrs_metrics = json.loads(open(f"{res_dir}/l_usrs_metrics.txt", "r").read())
        return (conf, losses, metrics, l_usrs_losses, l_usrs_metrics)
    else:
        return (conf, losses, metrics)

def plot_and_save(xs, ysm, ysf, title, save_path):
    plt.figure()
    plt.title(title)
    plt.xlabel("epoch")
    plt.plot(xs, ysm, label = "ML")
    plt.plot(xs, ysf, label = "FL")
    plt.legend()
    plt.savefig(save_path)


def plt_compare_figures(model, model_fl=""):
    if model_fl == "":
        model_fl = model

    conf, lsm, msm = load_res(f"./res/ML/{model}")
    conff, lsf, msf = load_res(f"./res/FL/{model_fl}")
    xs = [i + 1 for i in range(len(lsm))]

    # comparison of loss
    plot_and_save(xs, lsm, lsf, 
        "loss comparison",
        f"./imgs/{model_fl}/comparison-loss-{model_fl}.png"
    )

    # comparison acc
    plot_and_save(xs, accs(msm), accs(msf),
        "accuray comparison", 
        f"./imgs/{model_fl}/comparison-acc-{model_fl}.png"
    )

    # comparison pr
    plot_and_save(xs, prs(msm), prs(msf),
        "precison comparison", 
        f"./imgs/{model_fl}/comparison-pr-{model_fl}.png"
    )

    # comparison dr
    plot_and_save(xs, drs(msm), drs(msf),
        "detection rate comparison", 
        f"./imgs/{model_fl}/comparison-dr-{model_fl}.png"
    )

    # comparison fpr
    plot_and_save(xs, fprs(msm), fprs(msf),
        "fpr comparison", 
        f"./imgs/{model_fl}/comparison-fpr-{model_fl}.png"
    )

def plt_usr_records(xs, urs):
    for i in range(len(urs)):
        plt.plot(xs, urs[i], label = f"user {i}")
    plt.legend()