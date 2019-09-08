import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter


def GAP_evaluation(y_true, y_pred, scores):
    """
    Inputs:
    - y_true: true labels
    - y_pred: predicted labels
    - scores: scores (confidence) about prediction
    """
    N = len(y_true)
    whole = np.vstack((y_true, y_pred, scores)).T
    whole = whole[np.flip(whole[:, 2].argsort())]
    rel = (whole[:, 0] == whole[:, 1]).astype(float)
    p = np.cumsum(rel) / (np.arange(N) + 1)
    GAP = (p @ rel) / N
    return GAP


def check_accuracy(loader, model, mode, device=torch.device('cuda')):
    """
    Inputs:
    - mode: 'Siamese', 'Prototypical' or 'MetaSVM'
    """
    assert mode in ['Siamese', 'Prototypical', 'MetaSVM']
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)
            if mode in ['Prototypical', 'MetaSVM']:
                N_c, N_Q, C, W, H = x2.shape
                y = torch.Tensor(np.repeat(np.arange(N_c), N_Q)).to(device=device, dtype=torch.long)
                scores = model(x1, x2)
                _, preds = scores.max(1)
            else:
                y = y.to(device=device)
                scores = model(x1, x2)
                preds = (scores >= 0.5).view(-1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def test_Siamese(loader, model, device=torch.device('cuda')):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.nn.no_grad():
        for x1, x2, y in loader:
            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)
            N_c, N_S, C, W, H = x1.shape
            N_c, N_Q, C, W, H = x2.shape
            y = np.repeat(np.arange(N_c), N_Q)
            sup_label = np.repeat(np.arange(N_c), N_S)
            x1 = x1.view(-1, C, W, H)
            x2 = x2.view(-1, C, W, H)
            for i in range(N_c * N_Q):
                test_x = x2[i].repeat(N_c * N_S, 1, 1, 1)
                scores = model(x1, test_x)
                i_hat = scores.max(0)[1].item()
                preds = sup_label[i_hat]
                num_correct += (y[i] == preds)
                num_samples += 1
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc

def simple_train(model, train_loader, val_loader, optimizer, save_path, save_best_path,
                 scheduler=None, loss_func=torch.nn.BCELoss(), mode='Siamese',
                 print_interval=20, device=torch.device('cuda'), epochs=2):
    """
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - train_loader/ val_loader: Data loader for training/ validation set
    - optimizer: An Optimizer object we will use to train the model
    - loss_func: Loss function for optimizing
    - mode: 'Siamese' or 'Prototypical'
    - save_path: path to save the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    """
    assert mode in ['Siamese', 'Prototypical', 'MetaSVM']
    model = model.to(device=device)
    writer = SummaryWriter()
    itr = 0
    best_acc = 0.0
    for e in range(epochs):
        for t, (x1, x2, y) in enumerate(train_loader):
            model.train()
            x1 = x1.to(device=device, dtype=torch.float32)
            x2 = x2.to(device=device, dtype=torch.float32)
            if mode in ['Prototypical', 'MetaSVM']:
                N_c, N_Q, C, W, H = x2.shape
                y = torch.Tensor(np.repeat(np.arange(N_c), N_Q)).to(device=device, dtype=torch.long)
            else:
                y = y.to(device=device, dtype=torch.float32)

            scores = model(x1, x2)
            loss = loss_func(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss value', loss, itr)
            if mode in ['Prototypical', 'MetaSVM']:
                writer.add_image('sample image', x1[0, 0], itr)
            else:
                writer.add_image('sample image', x1[0], itr)

            if itr % print_interval == 0:
                print('Iteration %d, loss = %.4f' % (itr, loss.item()))
                if val_loader is not None:
                    val_acc = check_accuracy(val_loader, model, mode)
                    writer.add_scalar('validation accuracy', val_acc, itr)
                    print()
                state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, save_path)
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_state = {'state_dict': model.state_dict(
                    ), 'optimizer': optimizer.state_dict()}
                    torch.save(best_state, save_best_path)
            itr += 1

        if scheduler is not None:
            scheduler.step()
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, save_path)
    val_acc = check_accuracy(val_loader, model, mode)
    if val_acc >= best_acc:
        best_acc = val_acc
        best_state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(best_state, save_best_path)
    print('best acc is (%.2f)' % (100 * best_acc))
    writer.close()


def test_accuracy_proto(base_net, train_loader, test_loader, label_list, device=torch.device('cuda')):
    """
    Just used for general classification.
    """
    num_correct = 0
    num_samples = 0
    base_net.to(device=device)
    base_net.eval()
    with torch.no_grad():
        for i, (S_k, Q_k, y) in enumerate(train_loader):
            S_k = S_k.to(device=device)
            N_c, N_S, C, W, H = S_k.shape
            S_k = S_k.view(-1, C, W, H)
            c_k = torch.mean(base_net(S_k).view(N_c, N_S, -1), dim=1)
            print(f"iteration {i+1}.")
            if i == 0:
                c = c_k
            else:
                c = torch.cat((c, c_k), dim=0)
        c_norm = torch.sum(c * c, dim=1)
        print("prototype calculated!")
        for x, y in test_loader:
            x = x.to(device=device)
            y = y.view(-1)
            f_x = base_net(x).view(-1)
            dist = c_norm + torch.sum(f_x * f_x) - 2 * torch.matmul(c, f_x)
            preds = dist.min(0)[1].item()
            pred_label = label_list[preds]
            num_correct += int(pred_label == y)
            num_samples += 1
    acc = num_correct / num_samples
    print(acc)
    return acc
