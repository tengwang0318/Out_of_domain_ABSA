import csv

from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
import torch.nn.functional as F
import numpy as np
import torch
import time
from fastprogress.fastprogress import format_time, master_bar, progress_bar
from collections import defaultdict


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Taken from https://github.com/Bjarten/early-stopping-pytorch"""

    def __init__(self, filename, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.cur_date = filename

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves models when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving models ...')
        torch.save(model.state_dict(), 'models/' + self.cur_date + '_checkpoint.pt')
        self.val_loss_min = val_loss


class Trainer:
    """
    Class to encapsulate training and validation steps for a pipeline.
    """

    def __init__(self, model, train_data_loader, val_data_loader, file_name):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.file_name = file_name
        self.early_stop = EarlyStopping(self.file_name, patience=10)

    def fit(self, num_epochs, args, device='cuda:0'):
        """
        Fit the PyTorch models
        :param num_epochs: number of epochs to train (int)
        :param args:
        :param device: str (defaults to 'cuda:0')
        """
        optimizer, scheduler, step_scheduler_on_batch = self.optimizer(args)
        self.model = self.model.to(device)
        pbar = master_bar(range(num_epochs))
        headers = ['Train_Loss', 'Val_Loss', 'precision', 'recall', 'Time']
        pbar.write(headers, table=True)
        for epoch in pbar:
            epoch += 1
            start_time = time.time()
            self.model.train()
            overall_training_loss = 0.0
            for step, batch in enumerate(progress_bar(self.train_data_loader, parent=pbar)):
                loss, num_rows, _, _ = self.model(batch, device)
                overall_training_loss += loss.item() * num_rows

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                if step_scheduler_on_batch:
                    scheduler.step()
                optimizer.zero_grad()

            if not step_scheduler_on_batch:
                scheduler.step()

            overall_training_loss = overall_training_loss / len(self.train_data_loader.dataset)
            overall_val_loss, pred_dict = self.predict(device, pbar)
            y_true, y_pred = pred_dict['y_true'], pred_dict['y_pred']

            str_stats = []
            stats = [overall_training_loss,
                     overall_val_loss,
                     precision_score(y_true, y_pred,average='macro'),
                     recall_score(y_true, y_pred,average='macro'),
                     ]

            for stat in stats:
                str_stats.append(
                    'NA' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.4f}'
                )
            str_stats.append(format_time(time.time() - start_time))

            print(' '.join('{}: {}'.format(*k) for k in zip(headers, str_stats)))
            print('epoch#: ', epoch)
            pbar.write(str_stats, table=True)
            self.early_stop(overall_val_loss, self.model)
            if self.early_stop.early_stop:
                print("Early stopping")
                break

    def optimizer(self, args):
        """
        :param args: object
        """
        optimizer = AdamW([
            {'params': self.model.bert.parameters()},
            {'params': self.model.ffn.parameters(),
             'lr': float(args['--ffn-lr'])},
        ], lr=float(args['--bert-lr']), correct_bias=True)
        num_train_steps = (int(len(self.train_data_loader.dataset)) /
                           int(args['--train-batch-size'])) * int(args['--max-epoch'])
        num_warmup_steps = int(num_train_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_steps)
        step_scheduler_on_batch = True
        return optimizer, scheduler, step_scheduler_on_batch

    def predict(self, device='cuda:0', pbar=None):
        """
        Evaluate the models on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: overall_val_loss (float), accuracies (dict{'acc': value}, preds (dict)
        """
        current_size = len(self.val_data_loader.dataset)
        preds_dict = {
            'y_true': np.zeros([current_size, 1]),
            'y_pred': np.zeros([current_size, 1])
        }
        overall_val_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(progress_bar(self.val_data_loader, parent=pbar, leave=(pbar is not None))):
                loss, num_rows, y_pred, targets = self.model(batch, device)
                overall_val_loss += loss.item() * num_rows

                current_index = index_dict
                targets = np.reshape(targets, (num_rows, 1))
                y_pred = np.reshape(y_pred, (num_rows, 1))
                preds_dict['y_true'][current_index: current_index + num_rows, :] = targets
                preds_dict['y_pred'][current_index: current_index + num_rows, :] = y_pred
                index_dict += num_rows

        overall_val_loss = overall_val_loss / len(self.val_data_loader.dataset)

        return overall_val_loss, preds_dict


class EvaluateOnTest:
    def __init__(self, model, test_data_loader, model_path):
        self.model = model
        self.test_data_loader = test_data_loader
        self.model_path = model_path

    def predict(self, device='cuda:0', pbar=None):
        """
        Evaluate the models on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: None
        """
        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.test_data_loader.dataset)
        preds_dict = {
            'y_true': np.zeros([current_size, 3]),
            'y_pred': np.zeros([current_size, 3])
        }
        start_time = time.time()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(progress_bar(self.test_data_loader, parent=pbar, leave=(pbar is not None))):
                _, num_rows, y_pred, targets = self.model(batch, device)
                current_index = index_dict
                targets = np.reshape(targets, (num_rows, 1))
                y_pred = np.reshape(y_pred, (num_rows, 1))
                preds_dict['y_true'][current_index: current_index + num_rows, :] = targets
                preds_dict['y_pred'][current_index: current_index + num_rows, :] = y_pred
                index_dict += num_rows

        y_true, y_pred = preds_dict['y_true'], preds_dict['y_pred']
        str_stats = []
        stats = [precision_score(y_true, y_pred),
                 recall_score(y_true, y_pred), ]

        for stat in stats:
            str_stats.append(
                'NA' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.4f}'
            )

        str_stats.append(format_time(time.time() - start_time))
        headers = ["Precision", "Recall", 'Time']
        print(' '.join('{}: {}'.format(*k) for k in zip(headers, str_stats)))


class PredictTest:
    def __init__(self, model, fake_test_data_loader, real_test_data_loader, model_path):
        self.model = model
        self.fake_test_data_loader = fake_test_data_loader
        self.real_test_data_loader = real_test_data_loader
        self.model_path = model_path

    def predict(self, device='cuda:0', pbar=None):
        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.fake_test_data_loader.dataset)
        preds_dict = {
            'y_pred': np.zeros([current_size, 1])
        }
        start_time = time.time()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(progress_bar(self.fake_test_data_loader,
                                                      parent=pbar,
                                                      leave=(pbar is not None))):
                _, num_rows, y_pred, targets = self.model(batch, device)
                current_index = index_dict
                y_pred = np.reshape(y_pred, (num_rows, 1))
                preds_dict['y_pred'][current_index:current_index + num_rows, :] = y_pred
                index_dict += num_rows

        print(preds_dict['y_pred'].shape)
        with open('data/faker.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            fake_ids, fake_aspect_polarity, fake_sentences = [], [], []
            idx = 0
            for row in reader:
                if preds_dict['y_pred'][idx] == 1:
                    fake_ids.append(row[0])
                    fake_sentences.append(row[1])
                    fake_aspect_polarity.append(row[4])

                idx += 1

        with open("data/semEval2014.tsv") as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            ids, aspect_polarity, sentences = [], [], []
            for row in reader:
                if row[5] == 'yes':
                    ids.append(row[0])
                    aspect_polarity.append(row[4])
                    sentences.append(row[1])

        fake_data, data = defaultdict(list), defaultdict(list)
        # print(fake_aspect_polarity)
        # print(fake_ids)
        for i in range(len(fake_ids)):
            if fake_aspect_polarity[i] not in fake_data[fake_ids[i]]:
                fake_data[fake_ids[i]].append(fake_aspect_polarity[i])
        for i in range(len(ids)):
            if aspect_polarity[i] not in data[ids[i]]:
                data[ids[i]].append(aspect_polarity[i])

        print(data)
        print(fake_data)
        right = 0
        cnt = 0
        for key, aspects in fake_data.items():
            for asp in aspects:
                if asp in data[key]:
                    right += 1
                else:
                    print(asp)
                cnt += 1

        print(f"Precision: {right / cnt}")

        right, cnt = 0, 0
        for key, aspects in data.items():
            for asp in aspects:
                if asp in fake_data[key]:
                    right += 1
                else:
                    print(asp)
                cnt += 1
        print(f'recall: {right / cnt}')
        with open('data/realOOD.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'Sentence', 'aspect_polarity'])
            for _id, _sentence, _aspect_polarity in zip(ids, sentences, aspect_polarity):
                writer.writerow([_id, _sentence, _aspect_polarity])
        with open("data/predictOOD.tsv", 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'Sentence', "aspect_polarity"])
            for _id, _sentence, _aspect_polarity in zip(fake_ids, fake_sentences, fake_aspect_polarity):
                writer.writerow([_id, _sentence, _aspect_polarity])


class AllAspectsTest:
    def __init__(self, model, fake_test_data_loader, real_test_data_loader, model_path):
        self.model = model
        self.fake_test_data_loader = fake_test_data_loader
        self.real_test_data_loader = real_test_data_loader
        self.model_path = model_path

    def predict(self, device='cuda:0', pbar=None):
        """
        Evaluate the models on a validation set
        :param device: str (defaults to 'cuda:0')
        :param pbar: fast_progress progress bar (defaults to None)
        :returns: None
        """
        self.model.to(device).load_state_dict(torch.load(self.model_path))
        self.model.eval()
        current_size = len(self.fake_test_data_loader.dataset)
        preds_dict = {
            'y_true': np.zeros([current_size, 1]),
            'y_pred': np.zeros([current_size, 1])
        }
        start_time = time.time()
        with torch.no_grad():
            index_dict = 0
            for step, batch in enumerate(progress_bar(self.fake_test_data_loader, parent=pbar, leave=(pbar is not None))):
                _, num_rows, y_pred, targets = self.model(batch, device)
                current_index = index_dict
                targets = np.reshape(targets, (num_rows, 1))
                y_pred = np.reshape(y_pred, (num_rows, 1))
                preds_dict['y_true'][current_index: current_index + num_rows, :] = targets
                preds_dict['y_pred'][current_index: current_index + num_rows, :] = y_pred
                index_dict += num_rows

        with open('data/all_aspects_sentihood_test_fake.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            fake_ids, fake_aspect_polarity, fake_sentences = [], [], []
            idx = 0
            for row in reader:
                if preds_dict['y_pred'][idx] == 1:
                    fake_ids.append(row[0])
                    fake_sentences.append(row[1])
                    fake_aspect_polarity.append(row[4])

                idx += 1
        with open('data/all_aspects_sentihood_test.tsv') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            ids, aspect_polarity, sentences = [], [], []
            for row in reader:
                if row[5] == 'yes':
                    ids.append(row[0])
                    aspect_polarity.append(row[4])
                    sentences.append(row[1])

        fake_data, data = defaultdict(list), defaultdict(list)

        for i in range(len(fake_ids)):
            if fake_aspect_polarity[i] not in fake_data[fake_ids[i]]:
                fake_data[fake_ids[i]].append(fake_aspect_polarity[i])

        for i in range(len(ids)):
            if aspect_polarity[i] not in data[ids[i]]:
                data[ids[i]].append(aspect_polarity[i])

        right = 0
        cnt = 0
        for key, aspects in fake_data.items():
            for asp in aspects:
                if asp in data[key]:
                    right += 1
                # else:
                #     print(asp)
                cnt += 1

        print(f"Precision: {right / cnt}")

        right, cnt = 0, 0
        for key, aspects in data.items():
            for asp in aspects:
                if asp in fake_data[key]:
                    right += 1
                # else:
                #     print(asp)
                cnt += 1
        print(f'recall: {right / cnt}')
        with open('data/allAspectRealOOD.tsv', 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'Sentence', 'aspect_polarity'])
            for _id, _sentence, _aspect_polarity in zip(ids, sentences, aspect_polarity):
                writer.writerow([_id, _sentence, _aspect_polarity])
        with open("data/allAspectPredictOOD.tsv", 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['ID', 'Sentence', "aspect_polarity"])
            for _id, _sentence, _aspect_polarity in zip(fake_ids, fake_sentences, fake_aspect_polarity):
                writer.writerow([_id, _sentence, _aspect_polarity])
