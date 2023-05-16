import argparse
import logging
from load_data import *
from experiment import *


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--input_path", type=str, default="./input/out_file")
    parser.add_argument("--cpu", type=str, default="True")
    parser.add_argument("--train", type=str, default="True")
    parser.add_argument("--max_iterations", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=150)
    parser.add_argument("--output_size", type=int, default=50)
    parser.add_argument("--validate_every", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=50)

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

    if options["train"] == "True":

        logging.info("Starting train iterations")

        iterations = 0
        total_train_loss = 0
        best_TP = 0
        best_TN = 0

        while iterations < options["max_iterations"]:

            for data in train_loader:

                if iterations % options["print_every"] == 0:
                    logging.info(f"[ITERATION]: {iterations}")

                total_train_loss += experiment.train(data)

                if iterations % options["validate_every"] == 0:

                    true_pos, true_neg, false_pos, false_neg = experiment.validate(validation_loader)

                    logging.info(f"[VALIDATE] at iterations {iterations}")
                    logging.info(f'Probes belonging to the same device => True Positive: {true_pos:.2f}, '
                                 f'False Positive: {false_pos:.2f}')
                    logging.info(f'Probes belonging to different devices => True Negative: {true_neg:.2f}, '
                                 f'False Negative: {false_neg:.2f}')

                    if true_pos > best_TP:
                        logging.info("Saving checkpoint")
                        best_TP = true_pos
                        experiment.save_checkpoint(
                            f'{options["output_path"]}/best_checkpoint.pth',
                            iterations,
                            best_TP,
                            best_TN,
                            total_train_loss
                        )

                    if true_neg > best_TN:
                        logging.info("Saving checkpoint")
                        best_TN = true_neg
                        experiment.save_checkpoint(
                            f'{options["output_path"]}/best_checkpoint.pth',
                            iterations,
                            best_TP,
                            best_TN,
                            total_train_loss
                        )

                    experiment.save_checkpoint(
                        f'{options["output_path"]}/last_checkpoint.pth',
                        iterations,
                        best_TP,
                        best_TN,
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
