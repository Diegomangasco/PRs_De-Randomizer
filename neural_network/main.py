import argparse
import logging
from load_data import *
from experiment import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--input_path", type=str, default="./input/out_file.pcap")
    parser.add_argument("--cpu", type=str, default="True")
    parser.add_argument("--test", type=str, default="True")
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=40)
    parser.add_argument("--output_size", type=int, default=10)
    parser.add_argument("--validate_every", type=int, default=100)

    options = vars(parser.parse_args())

    return options


if __name__ == "__main__":
    options = parse_arguments()
    experiment = Experiment(
        options["hidden_size"],
        options["output_size"],
        options["alpha"],
        options["beta"],
        options["threshold"],
        options["learning_rate"],
        options["cpu"]
    )

    train_loader, validation_loader = load_data(options["input_path"], options["batch_size"])

    if options["test"] == "True":
        iterations = 0
        total_train_loss = 0
        best_recognition = 0

        while iterations < options["max_iterations"]:

            for data in train_loader:

                total_train_loss += experiment.train_iteration(data)

                if iterations % options["validate_every"] == 0:

                    logging.info(f"Validation at iterations {iterations}")

                    true_pos, true_neg, false_pos, false_neg = experiment.validate(validation_loader)

                    if true_pos + true_neg > best_recognition:
                        logging.info(f'[VALIDATE] True Positive: {true_pos:.2f}')
                        logging.info(f'[VALIDATE] True Negative: {true_neg:.2f}')
                        best_recognition = true_pos + true_neg
                        experiment.save_checkpoint(
                            f'{options["output_path"]}/best_checkpoint.pth',
                            iterations,
                            best_recognition,
                            total_train_loss
                        )

                    experiment.save_checkpoint(
                        f'{options["output_path"]}/last_checkpoint.pth',
                        iterations,
                        best_recognition,
                        total_train_loss
                    )

                iterations += 1
                if iterations > options["max_iterations"]:
                    break

    # experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    # true_pos, true_neg, false_pos, false_neg = experiment.validate(test_loader)
    # logging.info(f'[TEST] True Positive: {true_pos:.2f}')
    # logging.info(f'[TEST] True Negative: {true_neg:.2f}')
    # logging.info(f'[TEST] False Positive: {false_pos:.2f}')
    # logging.info(f'[TEST] False Negative: {false_neg:.2f}')
