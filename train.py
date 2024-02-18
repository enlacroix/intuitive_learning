import math
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as thfunc

from components import Decoder


def division_mod_p_data(p, eq_token, op_token):
    """
    x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p
    """
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = x * y % p

    # a◦b = c, “a”, “◦”, “b”, “=”, и “c” - отдельные токены.
    return torch.stack([x, op, y, eq, result])


def plus_mod_p_data(p, eq_token, op_token):
    # TODO объединить с предыдушей
    x = torch.arange(p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = x + y % p

    return torch.stack([x, op, y, eq, result])


def permutation_multiplication(x, y):
    assert x.size() == y.size(), "Размерности тензоров должны быть одинаковыми"
    result = torch.zeros_like(x)
    for i in range(x.size(0)):
        result[i] = y[x[i] - 1]

    return result


def permutation_invert(x):
    """
    result = [0] * len(self)
    for i in range(len(self)):
        result[self[i] - 1] = i + 1
    return Permutation(result)
    """
    result = torch.zeros_like(x)
    for i in range(x.size(0)):
        result[result[i] - 1] = i + 1

    return result


def perm_mult_data(p, eq_token, op_token):
    x = torch.arange(1, p)
    y = torch.arange(1, p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    result = permutation_multiplication(permutation_multiplication(x, y) , permutation_invert(x))

    return torch.stack([x, op, y, eq, result])


def run(args):
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eq_token = args.p
    op_token = args.p + 1

    # трансформер 2 layers, width 128, and 4 attention heads"
    model = Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
    ).to(device)

    data = perm_mult_data(args.p, eq_token, op_token)  # division_mod_p_data # операция деления по модулю 97, 50% в обучающую выборку

    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)

    train_data, valid_data = data[:, train_idx], data[:, valid_idx]

    # AdamW optimizer with lr (learning rate) 10−3, weight decay 1, β1 = 0.9, β2 = 0.98
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []

    for e in tqdm(range(int(args.budget) // steps_per_epoch)):

        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(train_data, True), (valid_data, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])

                    loss = thfunc.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]

            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])
        if (e + 1) % 100 == 0:
            TITLE = "Обучение операции xyx^-1 в группе S97"
            X_LABEL = "Количество шагов"
            LABEL_1 = "тренировочная выборка"
            LABEL_2 = "тестовая выборка"

            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            plt.plot(steps, train_acc, label=LABEL_1)
            plt.plot(steps, val_acc, label=LABEL_2)
            plt.legend()
            plt.title(TITLE)
            plt.xlabel(X_LABEL)
            plt.ylabel('Точность')
            plt.xscale("log", base=10)
            plt.savefig(f"rsrc/acc_{e + 1}.png", dpi=150)
            plt.close()

            plt.plot(steps, train_loss, label=LABEL_1)
            plt.plot(steps, val_loss, label=LABEL_2)
            plt.legend()
            plt.title(TITLE)
            plt.xlabel(X_LABEL)
            plt.ylabel('Функция потерь')
            plt.xscale("log", base=10)
            plt.savefig(f"rsrc/loss_{e + 1}.png", dpi=150)
            plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--p", type=int, default=97)  # 97
    parser.add_argument("--budget", type=int, default=3e3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--optimizer", default="Adam")
    model_args = parser.parse_args()
    run(model_args)
