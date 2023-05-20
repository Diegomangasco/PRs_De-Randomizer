import argparse
import logging
from load_data import *
from experiment import *

'''
HYPERPARAMETERS
max_iterations: between 500 and 7000 (step of 500 => 14 values), 
alpha: possible values (0.001, 0.01, 0.1, 1, 10, 100, 1000), 
beta: possible values (0.001, 0.01, 0.1, 1, 10, 100, 1000), 
threshold: between 0.5 and 2.5 (step 0.1 => 15 values), 
hidden_size: possible values (310, 300, 250, 200, 150, 100, 50, 25, 10), 
output_size: possible values (310, 300, 250, 200, 150, 100, 50, 25, 10) (must be <= hidden_size)

POSSIBLE SETS: 
14*7*7*15*45 = 463,050
'''

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_path", type=str, default=".")
    parser.add_argument("--input_path", type=str, default="./input/out_file")
    parser.add_argument("--cpu", type=str, default="True")
    parser.add_argument("--test", type=str, default="False")
    parser.add_argument("--train", type=str, default="True")
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--hidden_size", type=int, default=150)
    parser.add_argument("--output_size", type=int, default=50)
    parser.add_argument("--validate_every", type=int, default=50)
    parser.add_argument("--print_every", type=int, default=30)
    parser.add_argument("--fine_tuning_validation", type=str, default="False")

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
        best_result = 0

        while iterations < options["max_iterations"]:

            for data in train_loader:

                if iterations % options["print_every"] == 0:
                    logging.info(f"[ITERATION]: {iterations}")

                total_train_loss += experiment.train(data)

                if iterations % options["validate_every"] == 0 and options["fine_tuning_validation"] == "False":

                    true_pos, false_neg, true_neg, false_pos = experiment.validate(validation_loader)

                    logging.info(f"[VALIDATE] at iterations {iterations}")
                    logging.info(f'Probes belonging to the same device => True Positive: {true_pos:.2f}, '
                                 f'False Negative: {false_neg:.2f}')
                    logging.info(f'Probes belonging to different devices => False Positive: {false_pos:.2f}, '
                                 f'True Negative: {true_neg:.2f}')

                    if true_pos + true_neg > best_result:
                        logging.info("Saving checkpoint")
                        best_result = true_pos + true_neg
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

    if options["fine_tuning_validation"] == "True":
        true_pos, false_neg, true_neg, false_pos = experiment.validate(validation_loader)

        with open("./fine_tuning.txt", "w") as fp:
            fp.write("Accuracy (true positive + true negative): {} ({} + {}), iterations: {}, alpha: {}, beta: {}, hidden_size: {}, output_size: {}\n"
                    .format(true_pos + true_neg, true_pos, true_neg, options["max_iterations"], options["alpha"], options["beta"], options["hidden_size"], options["output_size"]))
    
    if options["test"] == "True":
        # Use last_checkpoint.pth since we train before with the optimal number of iterations coming from fine tuning process
        experiment.load_checkpoint(f'{options["output_path"]}/last_checkpoint.pth')
        ground_truth, count = experiment.test(test_loader)
        logging.info("[COUNT TESTING]")
        logging.info("Number of devices present in the .pcap file: {}\n".format(ground_truth))
        logging.info("Number of devices detected: {}\n".format(count))
        
