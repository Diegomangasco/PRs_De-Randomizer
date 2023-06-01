import torch
from statistics import mean
from model import *


class Experiment:

    # Hyper parameters: alpha, beta, threshold, hidden_size, output_size, learning_rate

    def __init__(self, features_number, hidden_size, output_size, alpha, beta, threshold, learning_rate, device):
        self.device = torch.device("cpu" if device else "cuda:0")

        # Setup model
        input_size = features_number
        self.model = ProbesEncoder(input_size, hidden_size, output_size)
        self.model.train()
        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
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

        ideal_output = list()
        real_output = list()
        for i in range(len(device_label)):
            for j in range(i, len(device_label)):
                distance = torch.dist(res[i], res[j]).item()
                if device_label[i] == device_label[j]:
                    ideal_output.append(1)
                    if distance <= self.threshold:
                        # Classified as similar (True Positive)
                        real_output.append(1)
                    else:
                        # Classified as different (False Negative)
                        real_output.append(0)
                else:
                    ideal_output.append(0)
                    if distance > self.threshold:
                        # Classified as different (True Negative)
                        real_output.append(0)
                    else:
                        # Classified as similar (False Positive)
                        real_output.append(1)

        ideal_output = torch.tensor(ideal_output, dtype=torch.float32, requires_grad=True)
        real_output = torch.tensor(real_output, dtype=torch.float32)
        loss = self.criterion(real_output, ideal_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, data):
        self.model.eval()
        ideal_output = list()
        real_output = list()
        accuracy_list = list()
        same_distance = list()
        different_distance = list()
        with torch.no_grad():
            for x, device_label in data:
                x = x.to(self.device)
                device_label = device_label.numpy()

                res = self.model(x)

                for i in range(len(device_label)):
                    for j in range(i, len(device_label)):
                        distance = torch.dist(res[i], res[j]).item()
                        if device_label[i] == device_label[j]:
                            # Probes belong to the same device
                            same_distance.append(distance)
                            ideal_output.append(1)
                            if distance <= self.threshold:
                                # Same device recognized
                                real_output.append(1)
                            else:
                                # Same device but not recognized
                                real_output.append(0)
                        else:
                            # Probes don't belong to the same device
                            different_distance.append(distance)
                            ideal_output.append(0)
                            if distance > self.threshold:
                                # Different devices recognized:
                                real_output.append(0)
                            else:
                                # Different devices not recognized
                                real_output.append(1)

                accuracy_list.append(
                    100 * sum(
                        [1 if real == ideal else 0 for real, ideal in zip(real_output, ideal_output)]
                    ) / len(
                        ideal_output
                    )
                )

                real_output.clear()
                ideal_output.clear()

        self.model.train()

        return mean(accuracy_list), mean(same_distance), mean(different_distance)

    def test(self, data):
        self.model.eval()
        clusters = dict()
        already_placed = set()

        with torch.no_grad():
            x = data.to(self.device)
            res = self.model(x)
            results = res.numpy()

            for i in range(len(results)):
                for j in range(i, len(results)):
                    # if the condition is satisfied, res[i] is very similar to res[j]
                    if i == j and i not in already_placed:
                        clusters[i] = 1
                    elif i in already_placed:
                        break
                    elif i != j and j not in already_placed and torch.dist(res[i], res[j]) <= self.threshold:
                        clusters[i] += 1  # Count new probe
                        already_placed.add(j)

            # Probes not grouped
            # for i in range(len(res)):
            #     if i not in already_placed:
            #         clusters[i] = 1

        return len(clusters.keys())

