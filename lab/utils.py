import random
import numpy as np

import copy
import datetime
from tqdm import tqdm

from typing import Optional, Callable, Type, Tuple, List, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BatchEncoding
from gensim.models import Word2Vec

from sklearn.metrics import precision_score, accuracy_score, recall_score

import re

TOKEN_RE = re.compile(r'[\w\d]+')


def tokenize_text_simple_regex(txt: str, min_token_size: int = 4) -> List[str]:
    all_tokens = TOKEN_RE.findall(txt.lower())
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize_corpus(texts: np.ndarray,
                    tokenizer: Callable[[str, Optional[int]], list] = tokenize_text_simple_regex,
                    min_token_size: int = 4) -> List[List[str]]:

    if tokenizer is tokenize_text_simple_regex:
        return [tokenizer(text, min_token_size) for text in texts]
    else:
        return [tokenizer(text) for text in texts]


def init_random_seed(value: int = 0) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True


def document_vectorizer(document: list, model: Type[Word2Vec]) -> np.ndarray:
    vectors = [model.wv[word] for word in document if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def count_parameters(model: Type[torch.nn.Module]) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EmotionDataset(Dataset):
    def __init__(self,
                 texts: np.ndarray,
                 tokenizer: BertTokenizer,
                 labels: list = None,
                 max_length: int = 50) -> None:
        super().__init__()

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[BatchEncoding, list]:
        text = self.texts[idx]

        encoded_text = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
        )

        if self.labels is not None:
            label = self.labels[idx]
            return encoded_text, label
        else:
            return encoded_text, []


def predict_with_model(model: nn.Module,
                       dataloader: DataLoader,
                       device: torch.device,
                       use_sigmoid: bool,
                       return_labels: bool) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    results_by_batch = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, total=len(dataloader), colour='green'):
            b_input_ids = batch_x['input_ids'].squeeze(1).to(device)
            b_attention_mask = batch_x['attention_mask'].squeeze(1).to(device)

            if return_labels:
                labels.append(batch_y.numpy())

            batch_pred = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask, labels=None).logits

            if use_sigmoid:
                batch_pred = torch.sigmoid(batch_pred)

            results_by_batch.append(batch_pred.detach().cpu().numpy())

    if return_labels:
        return np.concatenate(results_by_batch, 0), np.concatenate(labels, 0)
    else:
        return np.concatenate(results_by_batch, 0)


def train_eval_loop(model: nn.Module,
                    train_dataloader: DataLoader,
                    test_dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    epoch_n: int,
                    device: torch.device,
                    early_stopping_patience: int,
                    scheduler: int = None) -> Tuple[nn.Module, List[float], List[float]]:

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

        for batch_i, (batch_x, batch_y) in tqdm(enumerate(train_dataloader),
                                                total=len(train_dataloader),
                                                colour="green"):
            optimizer.zero_grad()

            b_input_ids = batch_x['input_ids'].squeeze(1).to(device)
            b_attention_mask = batch_x['attention_mask'].squeeze(1).to(device)
            b_labels = batch_y.to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask, labels=b_labels)

            loss = outputs.loss
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
                b_input_ids = batch_x['input_ids'].squeeze(1).to(device)
                b_attention_mask = batch_x['attention_mask'].squeeze(1).to(device)
                b_labels = batch_y.to(device)

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask, labels=b_labels)

                loss = outputs.loss

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

        if scheduler is not None:
            scheduler.step(mean_val_loss)
        print()

    return best_model, history_loss_train, history_loss_test


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.argmax(axis=1)

    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_recall, score_precision, score_acc]
