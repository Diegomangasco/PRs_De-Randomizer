import torch
import torch.nn as nn
from model import *


class Experiment:

    # Hyper parameters: alpha, beta, threshold, hidden_size, output_size, learning_rate

    def __init__(self, hidden_size, output_size, alpha, beta, threshold, learning_rate, device):
        self.device = torch.device("cpu" if device else "cuda:0")

        # Setup model
        input_size = 309
        self.model = ProbeEncoderDecoder(input_size, hidden_size, output_size)
        self.model.train()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = dict()

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data):
        x, device_label = data
        x = x.to(self.device)
        device_label = device_label.to(self.device)

        res = self.model(x)

        # loss_1 is computed for probes belonging to the same device
        # loss_2 is computed for probes belonging to different devices
        loss_1 = 0
        loss_2 = 0
        for i in range(device_label.shape[0]):
            for j in range(device_label.shape[0]):
                if torch.eq(device_label[i], device_label[j]).item():
                    loss_1 = loss_1 + self.criterion(res[i], x[j])
                else:
                    loss_2 = loss_2 + self.criterion(res[i], x[j])

        # loss_1 is a normal loss, loss_2 is an adversarial loss
        total_loss = self.alpha*loss_1 - self.beta*loss_2

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def validate(self, data):
        false_positive = 0
        true_positive = 0
        false_negative = 0
        true_negative = 0
        with torch.no_grad():
            for x, device_label in data:
                x = x.to(self.device)
                device_label = device_label.to(self.device)
                total = device_label.size(dim=0)

                res = self.model(x, train=False)

                for i in range(device_label.shape[0]):
                    for j in range(device_label.shape[0]):
                        if torch.eq(device_label[i], device_label[j]).item():
                            # Probes belong to the same device
                            if self.criterion(res[i], x[j]) < self.threshold:
                                # Same device recognized
                                true_positive += 1
                            else:
                                # Same device but not recognized
                                false_negative += 1
                        else:
                            # Probes don't belong to the same device
                            if self.criterion(res[i], x[j]) > self.threshold:
                                # Different devices recognized:
                                true_negative += 1
                            else:
                                # Different devices not recognized
                                false_positive += 1

        TP_ratio = 100 * (true_positive / total)
        TN_ratio = 100 * (true_negative / total)
        FP_ratio = 100 * (false_positive / total)
        FN_ratio = 100 * (false_negative / total)
        self.model.train()

        return TP_ratio, TN_ratio, FP_ratio, FN_ratio
