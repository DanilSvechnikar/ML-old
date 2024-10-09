import random
import numpy as np
import copy

from tqdm import tqdm

import torch
import torch.backends.cudnn
from torch.utils.tensorboard import SummaryWriter


def train_eval_loop_lstm(model,
                         train_dataloader, valid_dataloader,
                         optimizer, loss_fn,
                         epoch_n,
                         early_stopping_patience,
                         scheduler,
                         teacher_force_ratio):

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    writer = SummaryWriter(f'loss_plot')

    for epoch_i in range(epoch_n):
        print(f'Epoch [{epoch_i} / {epoch_n}]')

        model.train()
        mean_train_loss = 0
        train_batches_n = 0

        for batch_i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inp_data = batch.german
            target = batch.english

            output = model(inp_data, target, teacher_force_ratio)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = loss_fn(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            mean_train_loss += float(loss.item())
            train_batches_n += 1

        mean_train_loss /= train_batches_n
        writer.add_scalar('Loss/train', mean_train_loss, epoch_i)

        print('Average value of the learning loss function:', mean_train_loss)

        model.eval()
        mean_val_loss = 0
        val_batches_n = 0

        with torch.no_grad():
            for batch_i, batch in enumerate(valid_dataloader):
                inp_data = batch.german
                target = batch.english

                output = model(inp_data, target, teacher_force_ratio=0.0)

                output = output[1:].reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)

                loss = loss_fn(output, target)

                mean_val_loss += float(loss.item())
                val_batches_n += 1

        mean_val_loss /= val_batches_n
        writer.add_scalar('Loss/validation', mean_val_loss, epoch_i)

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

    return best_model


def train_eval_loop_attention(model,
                              train_dataloader, valid_dataloader,
                              optimizer, loss_fn,
                              epoch_n,
                              early_stopping_patience,
                              scheduler):

    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)

    writer = SummaryWriter(f'loss_plot_attention')

    for epoch_i in range(epoch_n):
        print(f'Epoch [{epoch_i} / {epoch_n}]')

        model.train()
        mean_train_loss = 0
        train_batches_n = 0

        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            inp_data = batch.german
            target = batch.english

            output = model(inp_data, target[:-1])

            output = output.reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()

            loss = loss_fn(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            mean_train_loss += float(loss.item())
            train_batches_n += 1

        mean_train_loss /= train_batches_n
        writer.add_scalar('Loss/train', mean_train_loss, epoch_i)

        print('Average value of the learning loss function:', mean_train_loss)

        model.eval()
        mean_val_loss = 0
        val_batches_n = 0

        with torch.no_grad():
            for batch_i, batch in enumerate(valid_dataloader):
                inp_data = batch.german
                target = batch.english

                output = model(inp_data, target[:-1])

                output = output.reshape(-1, output.shape[2])
                target = target[1:].reshape(-1)

                loss = loss_fn(output, target)

                mean_val_loss += float(loss.item())
                val_batches_n += 1

        mean_val_loss /= val_batches_n
        writer.add_scalar('Loss/validation', mean_val_loss, epoch_i)

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

    return best_model


def init_random_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_params_number(model):
    return sum(t.numel() for t in model.parameters())
