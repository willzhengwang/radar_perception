"""
In multiple classification, the loss function can be calculated in the following ways:

1. nll_loss(log_softmax(logits), labels)
2. cross

https://blog.csdn.net/zhangxb35/article/details/72464152?utm_source=itdadao&utm_medium=referral

Notes:
softmax(Z) = exp(Z) / sum(exp(Z)), Z is a vector if logits.

log_softmax(z) = log(softmax(z)).

log_softmax(z) is more efficient than log(softmax(z)). log_softmax can avoid overflow.

nll_loss(log_softmax(logits, dim=1), labels) == cross_entropy(logits, labels)

Time cost:

samples: 1000000, classes: 100
nll_loss:  0.22603225708007812
tensor(5.0955)
cross_entropy:  0.20372676849365234
tensor(5.0955)





"""
import time
import torch
from torch.nn.functional import log_softmax, softmax, nll_loss, cross_entropy


if __name__ == '__main__':
    # validation of the two ways of loss computation
    logits = torch.tensor([[1, -1, -1, 4, 0], [6, -1, -5, 1, 0]], dtype=torch.float32)
    print(softmax(logits, dim=1))
    print(log_softmax(logits, dim=1))

    labels = torch.tensor([3, 0], dtype=torch.int64)
    loss1 = nll_loss(log_softmax(logits, dim=1), labels)
    print(loss1)

    loss2 = cross_entropy(logits, labels)  #
    print(loss2)

    # compare the time cost
    n, k = 10000, 5
    print(f"samples: {n}, classes: {k}")
    logits = torch.randn([n, k], dtype=torch.float32)
    labels = torch.LongTensor(n).random_(k)
    start = time.time()
    loss1 = nll_loss(log_softmax(logits, dim=1), labels)
    end = time.time()
    print("nll_loss: ", end - start)
    print(loss1)

    start = time.time()

    loss2 = cross_entropy(logits, labels)
    # cross_entropy is same as torch.nn.CrossEntropyLoss().
    end = time.time()
    print("cross_entropy: ", end - start)
    print(loss2)



