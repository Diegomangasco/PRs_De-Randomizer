import argparse
from load_data import *
from experiment import *
from plot import *
import datetime

'''
HYPERPARAMETERS
max_iterations: always 7000 (saving best checkpoints and which iteration), 
alpha: possible values (0.001, 0.01, 0.1, 1, 10, 100, 1000), 
beta: possible values (0.001, 0.01, 0.1, 1, 10, 100, 1000), 
threshold: between 0.0001 and TO VERIFY (step 0.1 => 24 values), 
hidden_size: possible values (310, 250, 200, 150, 100, 50, 70, 10), 
output_size: possible values (310, 250, 200, 150, 100, 50, 70, 10) (must be <= hidden_size)

POSSIBLE SETS: 
7*7*24*36 = 42,336
'''


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--input_path", type=str, default="./input/train")
    parser.add_argument("--test_path", type=str, default="./input/test")
    parser.add_argument("--cpu", type=str, default="True")
    parser.add_argument("--test", type=str, default="False")
    parser.add_argument("--train", type=str, default="True")
    parser.add_argument("--max_iterations", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--hidden_size", type=int, default=150)
    parser.add_argument("--output_size", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=5)
    parser.add_argument("--validate_every", type=int, default=10)
    parser.add_argument("--fine_tuning", type=str, default="False")
    parser.add_argument("--graph", type=str, default="False")
    parser.add_argument("--figure_path", type=str, default="./graphs")

    options = vars(parser.parse_args())

    return options


if __name__ == "__main__":

    start_time = datetime.datetime.now()

    options = parse_arguments()

    train_loader, validation_loader, features_number = load_data(options["input_path"], options["batch_size"])

    experiment = Experiment(
        features_number,
        options["hidden_size"],
        options["output_size"],
        options["alpha"],
        options["beta"],
        options["threshold"],
        options["learning_rate"],
        options["cpu"]
    )

    if options["train"] == "True":

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

                    accuracy, same_distance, different_distance = experiment.validate(validation_loader)

                    if options["graph"] == "True":
                        with open("graphs/10_stats.txt", "a") as fp:
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
                        if options["fine_tuning"] == "True":
                            logging.info(
                                "Accuracy (true positive, true negative): {} ({}, {}), iterations: {}, alpha: {}, beta: {}, hidden_size: {}, output_size: {}, threshold: {}\n"
                                .format(best_result, true_pos, true_neg, iterations,
                                        options["alpha"],
                                        options["beta"], options["hidden_size"], options["output_size"],
                                        options["threshold"]))

                            with open("./fine_tuning.txt", "a") as fp:
                                fp.write(
                                    "Accuracy (true positive, true negative): {} ({}, {}), iterations: {}, alpha: {}, beta: {}, hidden_size: {}, output_size: {}, threshold: {}\n"
                                    .format(best_result, true_pos, true_neg, iterations,
                                            options["alpha"],
                                            options["beta"], options["hidden_size"], options["output_size"],
                                            options["threshold"]))

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
        test_loader, ground_truth = load_test(options["test_path"])
        experiment.load_checkpoint(f'{options["output_path"]}/best_checkpoint.pth')
        count = experiment.test(test_loader)
        logging.info("[COUNT TESTING]")
        logging.info("Number of devices present in the .pcap file: {}\n".format(ground_truth))
        logging.info("Number of devices detected: {}\n".format(count))
