import torch
from statistics import mean
from model import *
from collections import defaultdict


class Experiment:

    # Hyper parameters: alpha, beta, threshold, hidden_size, output_size, learning_rate

    def __init__(self, hidden_size, output_size, alpha, beta, threshold, learning_rate, device):
        self.device = torch.device("cpu" if device else "cuda:0")

        # Setup model
        input_size = 310  # Number of features
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

    def save_checkpoint(self, path, iteration, best_result, total_train_loss):
        checkpoint = dict()

        checkpoint["iteration"] = iteration
        checkpoint["best_result"] = best_result
        checkpoint["total_train_loss"] = total_train_loss
        checkpoint["model"] = self.model.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint["iteration"]
        best_result = checkpoint["best_result"]
        total_train_loss = checkpoint["total_train_loss"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        return iteration, best_result, total_train_loss

    def train(self, data):
        x, device_label = data
        x = x.to(self.device)
        device_label = device_label.numpy()
        res = self.model(x)

        # loss_1 is computed for probes belonging to the same device
        # loss_2 is computed for probes belonging to different devices
        loss_1 = 0
        loss_2 = 0
        total_positive_pairs = 0
        total_negative_pairs = 0
        for i in range(len(device_label)):
            for j in range(len(device_label)):
                if device_label[i] == device_label[j]:
                    total_positive_pairs += 1
                    loss_1 = loss_1 + self.criterion(res[i], res[j])
                else:
                    total_negative_pairs += 1
                    loss_2 = loss_2 + self.criterion(res[i], res[j])

        # loss_1 is a normal loss, loss_2 is an adversarial loss
        loss_1 /= total_positive_pairs
        loss_2 /= total_negative_pairs
        total_loss = self.alpha*loss_1 - self.beta*loss_2

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def validate(self, data):
        self.model.eval()
        false_positive = 0
        true_positive = 0
        false_negative = 0
        true_negative = 0
        total_positive = 0
        total_negative = 0
        TP_ratio = list()
        TN_ratio = list()
        FP_ratio = list()
        FN_ratio = list()
        with torch.no_grad():
            for x, device_label in data:
                x = x.to(self.device)
                device_label = device_label.numpy()

                res = self.model(x)

                for i in range(len(device_label)):
                    for j in range(len(device_label)):
                        if device_label[i] == device_label[j]:
                            # Probes belong to the same device
                            total_positive += 1
                            if self.criterion(res[i], res[j]) <= self.threshold:
                                # Same device recognized
                                true_positive += 1
                            else:
                                # Same device but not recognized
                                false_negative += 1
                        else:
                            # Probes don't belong to the same device
                            total_negative += 1
                            if self.criterion(res[i], res[j]) > self.threshold:
                                # Different devices recognized:
                                true_negative += 1
                            else:
                                # Different devices not recognized
                                false_positive += 1
                    TP_ratio.append(100 * (true_positive / total_positive))
                    FN_ratio.append(100 * (false_negative / total_positive))
                    TN_ratio.append(100 * (true_negative / total_negative))
                    FP_ratio.append(100 * (false_positive / total_negative))
                    total_negative = 0
                    total_positive = 0
                    true_positive = 0
                    true_negative = 0
                    false_negative = 0
                    false_positive = 0

        self.model.train()

        return mean(TP_ratio), mean(FN_ratio), mean(TN_ratio), mean(FP_ratio)

    def test(self, data):
        self.model.eval()
        clusters = dict()
        already_placed = set()

        with torch.no_grad():
            data = data.to(self.device)
            res = self.model(data)

            for i in range(len(res)):
                for j in range(len(res)):
                    # if the condition is satisfied, res[i] is very similar to res[j]
                    if i != j and j not in already_placed and self.criterion(res[i], res[j]) <= self.threshold:
                        if i not in clusters.keys():
                            clusters[i] = 1
                            already_placed.add(i)

                        clusters[i] += 1  # Count itself and the new one
                        already_placed.add(j)

            # Probes not grouped
            for i in range(len(res)):
                if i not in already_placed:
                    clusters[i] = 1

        return len(clusters.keys())

