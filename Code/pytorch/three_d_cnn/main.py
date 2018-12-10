import argparse

from time import time
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from classification_utils import *
from data_utils import *
from model import *

parser = argparse.ArgumentParser(description="Argparser for Pytorch 3D CNN")

# Mandatory arguments
parser.add_argument("diagnosis_tsv", type=str,
                    help="Path to tsv file of the population."
                         " To note, the column name should be participant_id, session_id and diagnosis.")
parser.add_argument("log_dir", type=str,
                    help="Path to log dir for tensorboard usage.")
parser.add_argument("input_dir", type=str,
                    help="Path to input dir of the MRI (preprocessed CAPS_dir).")
parser.add_argument("model", type=str, choices=["Conv_3", "Conv_4", "Test"],
                    help="model selected")

# Data Management
parser.add_argument("--batch_size", default=2, type=int,
                    help="Batch size for training. (default=1)")
parser.add_argument('--accumulation_steps', '-asteps', default=1, type=int,
                    help='Accumulates gradients in order to increase the size of the batch')
parser.add_argument("--shuffle", default=True, type=bool,
                    help="Load data if shuffled or not, shuffle for training, no for test data.")
parser.add_argument("--runs", default=1, type=int,
                    help="Number of runs with the same training / validation split.")
parser.add_argument("--test_sessions", default=["ses-M00"], nargs='+', type=str,
                    help="Test the accuracy at the end of the model for the sessions selected")
parser.add_argument("--visualization", default=None, type=str,
                    help='the file path to perform a test for autoencoder visualization')
parser.add_argument("--num_workers", '-w', default=1, type=int,
                    help='the number of batch being loaded in parallel')

# Pretraining arguments
parser.add_argument("-t", "--transfer_learning", default=False, action='store_true',
                    help="If do transfer learning")
parser.add_argument("--transfer_learning_tsv", "-t_tsv", type=str, default=None,
                    help='If transfer learning, gives the tsv file to use to perform pretraining')
parser.add_argument("--transfer_learning_epochs", "-t_e", type=int, default=10,
                    help="Number of epochs for pretraining")

# Training arguments
parser.add_argument("--epochs", default=20, type=int,
                    help="Epochs through the data. (default=20)")
parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float,
                    help="Learning rate of the optimization. (default=0.01)")
parser.add_argument("--tolerance", "-tol", default=5e-2, type=float,
                    help="Allows to stop when the training data is nearly learnt")

# Optimizer arguments
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adadelta", "Adam"],
                    help="Optimizer of choice for training. (default=Adam)")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--gpu', action='store_true', default=False,
                    help='Uses gpu instead of cpu if cuda is available')
parser.add_argument('--evaluation_steps', '-esteps', default=1, type=int,
                    help='Fix the number of batches to use before validation')
parser.add_argument('--num_threads', type=int, default=1,
                    help='Number of threads used.')


def main(options):

    check_and_clean(options.log_dir)
    torch.set_num_threads(options.num_threads)
    valid_accuracies = np.zeros(options.runs)
    if options.evaluation_steps % options.accumulation_steps != 0 and options.evaluation_steps != 1:
        raise Exception('Evaluation steps %d must be a multiple of accumulation steps %d' %
                        (options.evaluation_steps, options.accumulation_steps))

    transformations = None

    total_time = time()
    # Pretraining the model
    if options.transfer_learning:
        model = eval(options.model)()
        criterion = torch.nn.MSELoss()
        if options.transfer_learning is None:
            raise Exception("A tsv file with data for pretraining must be given")
        training_tsv, valid_tsv = load_split(options.transfer_learning_tsv)

        data_train = MRIDataset(options.input_dir, training_tsv, transformations)
        data_valid = MRIDataset(options.input_dir, valid_tsv, transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=options.shuffle,
                                  num_workers=options.num_workers,
                                  drop_last=True
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False
                                  )

        ae_pretraining(model, train_loader, valid_loader, criterion, True, options)

    for run in range(options.runs):
        # Get the data.
        training_tsv, valid_tsv = load_split(options.diagnosis_tsv)

        data_train = MRIDataset(options.input_dir, training_tsv, transform=transformations)
        data_valid = MRIDataset(options.input_dir, valid_tsv, transform=transformations)

        # Use argument load to distinguish training and testing
        train_loader = DataLoader(data_train,
                                  batch_size=options.batch_size,
                                  shuffle=options.shuffle,
                                  num_workers=options.num_workers,
                                  drop_last=True
                                  )

        valid_loader = DataLoader(data_valid,
                                  batch_size=options.batch_size,
                                  shuffle=False,
                                  num_workers=options.num_workers,
                                  drop_last=False
                                  )

        # Initialize the model
        model = create_model(options)

        # Define criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = eval("torch.optim." + options.optimizer)(filter(lambda x: x.requires_grad, model.parameters()),
                                                             options.learning_rate)

        training_time = time()
        train(model, train_loader, valid_loader, criterion, optimizer, run, options)
        training_time = time() - training_time

        # Load best model
        best_model, best_epoch = load_model(model, os.path.join(options.log_dir, "run" + str(run)))

        # Get best performance
        acc_mean_train_subject = test(best_model, train_loader, options.gpu)
        acc_mean_valid_subject = test(best_model, valid_loader, options.gpu)
        accuracies = (acc_mean_train_subject, acc_mean_valid_subject)
        write_summary(options.log_dir, run, accuracies, best_epoch, training_time)

    total_time = time() - total_time
    print("Total time of computation: %d s" % total_time)
    text_file = open(path.join(options.log_dir, 'model_output.txt'), 'w')
    text_file.write('Time of training: %d s \n' % total_time)
    text_file.write('Mean best validation accuracy: %.2f %% \n' % np.mean(valid_accuracies))
    text_file.write('Standard variation of best validation accuracy: %.2f %% \n' % np.std(valid_accuracies))
    text_file.close()


def write_summary(log_dir, run, accuracies, best_epoch, time):
    fold_dir = path.join(log_dir, "run" + str(run))
    text_file = open(path.join(fold_dir, 'run_output.txt'), 'w')
    text_file.write('Fold: %i \n' % run)
    text_file.write('Best epoch: %i \n' % best_epoch)
    text_file.write('Time of training: %d s \n' % time)
    text_file.write('Accuracy on training set: %.2f %% \n' % accuracies[0])
    text_file.write('Accuracy on validation set: %.2f %% \n' % accuracies[1])
    text_file.close()


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        print("unknown arguments: %s" % parser.parse_known_args()[1])
    main(options)
