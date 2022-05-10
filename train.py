import torch
from functools import *

def train(net, loss_func, optimizer, train_dl, device):
    #定义损失函数
    net.train()
    loss, correct, total, batch_total = 0.0, 0.0, 0, 0

    for (_, (data, label)) in enumerate(train_dl):
        batch_total += 1

        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(data)
        l = loss_func(output, label).sum()
        l.backward()
        optimizer.step()
        loss += l.item()
        correct += (output.argmax(dim = 1) == label).sum().item()
        total += label.shape[0]

    loss /= batch_total
    acc = correct / total
    print(f"Loss: {loss:.3f} | Acc: {100.0 * acc:.3f}% ({correct}/{total})")
    return loss, acc

def test(net, test_dl, device):
    ce_loss = torch.nn.CrossEntropyLoss()
    net.eval()
    loss, correct, total, batch_total = 0.0, 0, 0, 0
    pred_stat = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_dl):
            data, label = data.to(device), label.to(device)
            output = net(data)
            l = ce_loss(output, label).sum()
            loss += l.item()
            output = output.argmax(dim = 1)
            pred_stat += list(zip(
                map(lambda x : int(x), label), 
                map(lambda x : int(x), output)))
            # correct += (out == label).sum().item()
            # total += label.shape[0]
            batch_total += 1
    loss /= batch_total
    # acc = correct / total
    # print(f"Loss: {loss:.3f} | Acc: {100.0 * acc:.3f}% ({correct}/{total})")
    # return loss, acc
    return loss, pred_stat
