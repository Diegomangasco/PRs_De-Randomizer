import argparse
import logging

from load_data import *
from experiment import *
from plot import *
import datetime


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--input_path", type=str, default="./input/train")
    parser.add_argument("--test_path", type=str, default="./input/test")
    parser.add_argument("--cpu", type=str, default="True")
    parser.add_argument("--test", type=str, default="False")
    parser.add_argument("--train", type=str, default="True")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--hidden_size", type=int, default=250)
    parser.add_argument("--output_size", type=int, default=150)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--validate_every", type=int, default=20)
    parser.add_argument("--fine_tuning", type=str, default="False")
    parser.add_argument("--graph", type=str, default="False")
    parser.add_argument("--figure_path", type=str, default="./graphs")

    options = vars(parser.parse_args())

    return options


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    logging.getLogger().setLevel(logging.INFO)

    options = parse_arguments()

    experiment = None

    if options["train"] == "True":

        train_loader, validation_loader, features = load_data(options["input_path"], options["batch_size"])

        with open("./features.txt", "w") as fw:
            for f in features:
                fw.write("{} ".format(f))
        print("stop")
        experiment = Experiment(
            features,
            options["hidden_size"],
            options["output_size"],
            options["threshold"],
            options["learning_rate"],
            options["cpu"]
        )

        logging.info("Starting train iterations")

        iterations = 0
        total_train_loss = 0
        best_result = 0

        while iterations < options["max_iterations"]:

            for data in train_loader:

                total_train_loss += experiment.train(data)

                if iterations % options["print_every"] == 0:
                    logging.info(f"[ITERATION]: {iterations}, Loss: {total_train_loss}")

                if iterations % options["validate_every"] == 0:

                    accuracy = experiment.validate(validation_loader)

                    if options["graph"] == "True":
                        with open("graphs/stats.txt", "a") as fp:
                            fp.write("{} {} {} {}\n".format(iterations, accuracy, same_distance, different_distance))

                    logging.info(f"[VALIDATE] at iterations {iterations}")
                    logging.info(f"Accuracy on the validation set: {accuracy}")

                    if accuracy > best_result:
                        logging.info("Saving checkpoint")
                        best_result = accuracy
                        experiment.save_checkpoint(
                            f'{options["output_path"]}/best_checkpoint.pth',
                            iterations,
                            best_result,
                            total_train_loss
                        )

                    experiment.save_checkpoint(
                        f'{options["output_path"]}/last_checkpoint.pth',
                        iterations,
                        best_result,
                        total_train_loss
                    )

                iterations += 1
                if iterations > options["max_iterations"]:
                    break

        total_train_time = (datetime.datetime.now() - start_time).seconds / 60
        logging.info(
            f"End train iterations, total time = {total_train_time} minutes"
        )

        if options["graph"] == "True":
            graph_plot(options["figure_path"], options["threshold"])

    if options["test"] == "True":
        # Use last_checkpoint.pth since we train before with the optimal number of iterations coming from fine-tuning process
        if experiment is None:
            features = list()
            with open("./features.txt", "r") as fr:
                line = fp.readline()
                elements = line.split(" ")[:-1]
                for e in elements:
                    if any(c == "[" or c == "]" or c == "\n" for c in e):
                        e.replace("[", "")
                        e.replace("]", "")
                        e.replace("\n", "")
                        features.append(e)

            experiment = Experiment(
                features,
                options["hidden_size"],
                options["output_size"],
                options["threshold"],
                options["learning_rate"],
                options["cpu"]
            )
        logging.info("Loading checkpoint")
        experiment.load_checkpoint(f'{options["output_path"]}/last_checkpoint.pth')
        logging.info("Loading test file")
        test_loader, ground_truth = load_test(options["test_path"], experiment.features)
        logging.info("Starting test")
        count_greedy, count_dbscan = experiment.test(test_loader)
        logging.info("[COUNT TESTING]")
        logging.info("Number of devices present in the .pcap file: {}\n".format(ground_truth))
        logging.info("Number of devices detected with greedy clustering: {}\n".format(count_greedy))
        logging.info("Number of devices detected with DBSCAN clustering: {}\n".format(count_dbscan))
        total_train_time = (datetime.datetime.now() - start_time).seconds / 60
        logging.info(
            f"End test, total time = {total_train_time} minutes"
        )
