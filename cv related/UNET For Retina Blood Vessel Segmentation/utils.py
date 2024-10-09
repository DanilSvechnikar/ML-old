import random
import numpy as np

import torch
import torch.backends.cudnn

import copy
import datetime
from tqdm import tqdm

from sklearn.metrics import jaccard_score, \
    f1_score, recall_score, precision_score, \
    accuracy_score


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=2)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


def calculate_metrics(y_true, y_pred):
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def predict_with_model(model, dataloader, device, use_sigmoid, return_labels):
    results_by_batch = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, total=len(dataloader)):
            batch_x = batch_x.to(device, dtype=torch.float32)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(batch_x)

            if use_sigmoid:
                batch_pred = torch.sigmoid(batch_pred)

            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)


def train_eval_loop(model,
                    train_dataloader, test_dataloader,
                    optimizer, loss_fn,
                    epoch_n, device,
                    early_stopping_patience,
                    scheduler):

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    history_loss_train = list()
    history_loss_test = list()

    for epoch_i in range(epoch_n):
        epoch_start = datetime.datetime.now()
        print('Epoch {}'.format(epoch_i))

        """ Training the model """
        model.train()
        mean_train_loss = 0
        train_batches_n = 0

        for batch_i, (batch_x, batch_y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()

            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)

            pred = model(batch_x)

            loss = loss_fn(pred, batch_y)
            loss.backward()

            optimizer.step()

            mean_train_loss += float(loss.item())
            train_batches_n += 1

        mean_train_loss /= train_batches_n
        history_loss_train.append(mean_train_loss)

        epoch_end = datetime.datetime.now()
        print('Epoch: {} iterations, {:0.2f}s'.format(train_batches_n,
                                                      (epoch_end - epoch_start).total_seconds()))
        print('Average value of the learning loss function:', mean_train_loss)

        """ Evaluate the model """
        model.eval()
        mean_val_loss = 0
        val_batches_n = 0

        with torch.no_grad():
            for batch_i, (batch_x, batch_y) in enumerate(test_dataloader):
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device, dtype=torch.float32)

                pred = model(batch_x)

                loss = loss_fn(pred, batch_y)

                mean_val_loss += float(loss.item())
                val_batches_n += 1

        mean_val_loss /= val_batches_n
        history_loss_test.append(mean_val_loss)

        print('Average value of the validation loss function:', mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_epoch_i = epoch_i
            best_val_loss = mean_val_loss
            best_model = copy.deepcopy(model)
            print('The new best model!')
        elif epoch_i - best_epoch_i > early_stopping_patience:
            print('The model has not improved over the last {} epochs, stop learning!'.format(
                early_stopping_patience))
            break

        scheduler.step(mean_val_loss)
        print()

    return best_model, history_loss_train, history_loss_test


def init_random_seed(seed=123):
    """ Seeding the randomness """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_params_number(model):
    return sum(t.numel() for t in model.parameters())
