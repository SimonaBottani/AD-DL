# coding: utf8

import torch
import numpy as np
import os
import warnings
import pandas as pd
from time import time

from clinicadl.tools.deep_learning.iotools import check_and_clean
from clinicadl.tools.deep_learning import EarlyStopping, save_checkpoint


#####################
# CNN train / test  #
#####################

def train(model, train_loader, valid_loader, criterion, optimizer, resume, log_dir, model_dir,
          model_name, options):
    """
    Function used to train a CNN.
    The best model and checkpoint will be found in the 'best_model_dir' of options.output_dir.

    Args:
        model: (Module) CNN to be trained
        train_loader: (DataLoader) wrapper of the training dataset
        valid_loader: (DataLoader) wrapper of the validation dataset
        criterion: (loss) function to calculate the loss
        optimizer: (torch.optim) optimizer linked to model parameters
        resume: (bool) if True, a begun job is resumed
        log_dir: (str) path to the folder containing the logs
        model_dir: (str) path to the folder containing the models weights and biases
        options: (Namespace) ensemble of other options given to the main script.
    """
    print('multiclass training is: ' + str(options.multiclass))

    from tensorboardX import SummaryWriter
    from time import time

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(log_dir)

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'validation'))

    # Create tsv
    columns = ['epoch', 'iteration', 'bacc_train', 'mean_loss_train', 'bacc_valid', 'mean_loss_valid']
    filename = os.path.join(log_dir, 'training.tsv')

    # Initialize variables
    best_valid_accuracy = 0.0
    best_valid_loss = np.inf
    epoch = options.beginning_epoch



    model.train()  # set the module to training mode

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    mean_loss_valid = None

    while epoch < options.epochs and not early_stopping.step(mean_loss_valid):
        print("At %d-th epoch." % epoch)

        model.zero_grad()
        evaluation_flag = True
        step_flag = True
        tend = time()
        total_time = 0

        for i, data in enumerate(train_loader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if options.gpu:
                imgs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                imgs, labels = data['image'], data['label']

            ### to delete after test
            imgs[imgs != imgs] = 0
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

            if model_name == 'GoogLeNet3D_new':
                train_output, out_1, out_2 = model(imgs)
                _, predict_batch = train_output.topk(1)
                _, predict_batch_1 = out_1.topk(1)
                _, predict_batch_2 = out_2.topk(1)
                loss_1 = criterion(train_output, labels)
                loss_2 = criterion(out_1, labels)
                loss_3 = criterion(out_2, labels)
                loss = loss_1 + loss_2 + loss_3
            else:
                train_output = model(imgs)
                _, predict_batch = train_output.topk(1)
                loss = criterion(train_output, labels)

            # Back propagation
            loss.backward()

            del imgs, labels

            if (i + 1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                del loss

                # Evaluate the model only when no gradients are accumulated
                if options.evaluation_steps != 0 and (i + 1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)

                    if options.multiclass == False:
                        _, results_train = test(model, train_loader, options.gpu, criterion)
                        _, results_valid = test(model, valid_loader, options.gpu, criterion)
                    elif options.multiclass == True:
                        _, results_train = test(model, train_loader, options.gpu, criterion, multiclass=True)
                        _, results_valid = test(model, valid_loader, options.gpu, criterion, multiclass=True)


                    mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)
                    mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
                    model.train()

                    global_step = i + epoch * len(train_loader)
                    writer_train.add_scalar('balanced_accuracy', results_train["balanced_accuracy"], global_step)
                    writer_train.add_scalar('loss', mean_loss_train, global_step)
                    writer_valid.add_scalar('balanced_accuracy', results_valid["balanced_accuracy"], global_step)
                    writer_valid.add_scalar('loss', mean_loss_valid, global_step)

                    # Write results on the dataframe
                    row = np.array([epoch, i, results_train["balanced_accuracy"], mean_loss_train, results_valid["balanced_accuracy"], mean_loss_valid]).reshape(1, -1)
                    row_df = pd.DataFrame(row, columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

                    print("%s level training accuracy is %f at the end of iteration %d"
                          % (options.mode, results_train["balanced_accuracy"], i))
                    print("%s level validation accuracy is %f at the end of iteration %d"
                          % (options.mode, results_valid["balanced_accuracy"], i))

            tend = time()
        print('Mean time per batch loading (train):', total_time / len(train_loader) * train_loader.batch_size)

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        elif evaluation_flag and options.evaluation_steps != 0:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        model.zero_grad()
        print('Last checkpoint at the end of the epoch %d' % epoch)

        if options.multiclass == False:
            _, results_train = test(model, train_loader, options.gpu, criterion)
            _, results_valid = test(model, valid_loader, options.gpu, criterion)
        elif options.multiclass == True:
            _, results_train = test(model, train_loader, options.gpu, criterion, multiclass=True)
            _, results_valid = test(model, valid_loader, options.gpu, criterion, multiclass=True)


        mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)
        mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
        model.train()

        global_step = (epoch + 1) * len(train_loader)
        writer_train.add_scalar('balanced_accuracy', results_train["balanced_accuracy"], global_step)
        writer_train.add_scalar('loss', mean_loss_train, global_step)
        writer_valid.add_scalar('balanced_accuracy', results_valid["balanced_accuracy"], global_step)
        writer_valid.add_scalar('loss', mean_loss_valid, global_step)

        # Write results on the dataframe
        row = np.array([epoch, i, results_train["balanced_accuracy"], mean_loss_train, results_valid["balanced_accuracy"],
             mean_loss_valid]).reshape(1, -1)
        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')

        print("%s level training accuracy is %f at the end of iteration %d"
              % (options.mode, results_train["balanced_accuracy"], len(train_loader)))
        print("%s level validation accuracy is %f at the end of iteration %d"
              % (options.mode, results_valid["balanced_accuracy"], len(train_loader)))

        accuracy_is_best = results_valid["balanced_accuracy"] > best_valid_accuracy
        loss_is_best = mean_loss_valid < best_valid_loss
        best_valid_accuracy = max(results_valid["balanced_accuracy"], best_valid_accuracy)
        best_valid_loss = min(mean_loss_valid, best_valid_loss)

        save_checkpoint({'model': model.state_dict(),
                         'epoch': epoch,
                         'valid_loss': mean_loss_valid,
                         'valid_acc': results_valid["balanced_accuracy"]},
                        accuracy_is_best, loss_is_best,
                        model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': options.optimizer,
                         },
                        False, False,
                        model_dir,
                        filename='optimizer.pth.tar')

        epoch += 1

    os.remove(os.path.join(model_dir, "optimizer.pth.tar"))
    os.remove(os.path.join(model_dir, "checkpoint.pth.tar"))


def evaluate_prediction(y, y_pred):
    """
    Evaluates different metrics based on the list of true labels and predicted labels.

    Args:
        y: (list) true labels
        y_pred: (list) corresponding predictions

    Returns:
        (dict) ensemble of metrics
    """

    true_positive = np.sum((y_pred == 1) & (y == 1))
    true_negative = np.sum((y_pred == 0) & (y == 0))
    false_positive = np.sum((y_pred == 1) & (y == 0))
    false_negative = np.sum((y_pred == 0) & (y == 1))

    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    if (true_positive + false_negative) != 0:
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        sensitivity = 0.0

    if (false_positive + true_negative) != 0:
        specificity = true_negative / (false_positive + true_negative)
    else:
        specificity = 0.0

    if (true_positive + false_positive) != 0:
        ppv = true_positive / (true_positive + false_positive)
    else:
        ppv = 0.0

    if (true_negative + false_negative) != 0:
        npv = true_negative / (true_negative + false_negative)
    else:
        npv = 0.0

    balanced_accuracy = (sensitivity + specificity) / 2

    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy,
               'sensitivity': sensitivity,
               'specificity': specificity,
               'ppv': ppv,
               'npv': npv,
               }

    return results

def evaluate_prediction_multiclass(y, ypred):
    """
    Evaluates different metrics based on the list of true labels and predicted labels for multiclass classification.

    Args:
        y: (list) true labels
        y_pred: (list) corresponding predictions

    Returns:
        (dict) ensemble of metrics
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    balanced_accuracy = balanced_accuracy_score(y, ypred)
    accuracy = accuracy_score(y, ypred)
    results = {'accuracy': accuracy,
               'balanced_accuracy': balanced_accuracy
               }
    return results



def test(model, dataloader, use_cuda, criterion, mode="image", multiclass=False):
    """
    Computes the predictions and evaluation metrics.

    Args:
        model: (Module) CNN to be tested.
        dataloader: (DataLoader) wrapper of a dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
    Returns
        (DataFrame) results of each input.
        (dict) ensemble of metrics + total loss on mode level.
    """
    print('multiclass is')
    print(multiclass)
    model.eval() ##automatically, self.training=False

    if mode == "image":
        columns = ["participant_id", "session_id", "true_label", "predicted_label"]
    elif mode in ["patch", "roi", "slice"]:
        columns = ['participant_id', 'session_id', '%s_id' % mode, 'true_label', 'predicted_label', 'proba0', 'proba1']
    else:
        raise ValueError("The mode %s is invalid." % mode)

    softmax = torch.nn.Softmax(dim=1)
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0
    total_time = 0
    tend = time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if use_cuda:
                inputs, labels = data['image'].cuda(), data['label'].cuda()
            else:
                inputs, labels = data['image'], data['label']


            inputs[inputs != inputs] = 0
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())


            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                if mode == "image":
                    row = [[sub, data['session_id'][idx], labels[idx].item(), predicted[idx].item()]]
                else:
                    normalized_output = softmax(outputs)
                    row = [[sub, data['session_id'][idx], data['%s_id' % mode][idx].item(),
                            labels[idx].item(), predicted[idx].item(),
                            normalized_output[idx, 0].item(), normalized_output[idx, 1].item()]]

                row_df = pd.DataFrame(row, columns=columns)
                results_df = pd.concat([results_df, row_df])

            del inputs, outputs, labels, loss
            tend = time()
        print('Mean time per batch loading (test):', total_time / len(dataloader) * dataloader.batch_size)
        results_df.reset_index(inplace=True, drop=True)

        # calculate the balanced accuracy
        if multiclass == False:
            results = evaluate_prediction(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))
        elif multiclass == True:
            results = evaluate_prediction_multiclass(results_df.true_label.values.astype(int),
                                      results_df.predicted_label.values.astype(int))


        results_df.reset_index(inplace=True, drop=True)
        results['total_loss'] = total_loss
        torch.cuda.empty_cache()



    return results_df, results


#################################
# Voting systems
#################################

def mode_level_to_tsvs(output_dir, results_df, metrics, fold, selection, mode, dataset='train', cnn_index=None):
    """
    Writes the outputs of the test function in tsv files.

    Args:
        output_dir: (str) path to the output directory.
        results_df: (DataFrame) the individual results per patch.
        metrics: (dict or DataFrame) the performances obtained on a series of metrics.
        fold: (int) the fold for which the performances were obtained.
        selection: (str) the metrics on which the model was selected (best_acc, best_loss)
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
        dataset: (str) the dataset on which the evaluation was performed.
        cnn_index: (int) provide the cnn_index only for a multi-cnn framework.
    """
    if cnn_index is None:
        performance_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
    else:
        performance_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', 'cnn-%i' % cnn_index,
                                       selection)
        metrics["%s_id" % mode] = cnn_index

    if not os.path.exists(performance_dir):
        os.makedirs(performance_dir)

    results_df.to_csv(os.path.join(performance_dir, '%s_%s_level_prediction.tsv' % (dataset, mode)), index=False,
                      sep='\t')

    if isinstance(metrics, dict):
        pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode)),
                                                index=False, sep='\t')
    elif isinstance(metrics, pd.DataFrame):
        metrics.to_csv(os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode)),
                       index=False, sep='\t')
    else:
        raise ValueError("Bad type for metrics: %s. Must be dict or DataFrame." % type(metrics).__name__)


def concat_multi_cnn_results(output_dir, fold, selection, mode, dataset, num_cnn):
    """Concatenate the tsv files of a multi-CNN framework"""
    prediction_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    for cnn_index in range(num_cnn):
        cnn_dir = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', 'cnn-%i' % cnn_index)
        performance_dir = os.path.join(cnn_dir, selection)
        cnn_pred_path = os.path.join(performance_dir, '%s_%s_level_prediction.tsv' % (dataset, mode))
        cnn_metrics_path = os.path.join(performance_dir, '%s_%s_level_metrics.tsv' % (dataset, mode))

        cnn_pred_df = pd.read_csv(cnn_pred_path, sep='\t')
        cnn_metrics_df = pd.read_csv(cnn_metrics_path, sep='\t')
        prediction_df = pd.concat([prediction_df, cnn_pred_df])
        metrics_df = pd.concat([metrics_df, cnn_metrics_df])

        # Clean unused files
        os.remove(cnn_pred_path)
        os.remove(cnn_metrics_path)
        if len(os.listdir(performance_dir)) == 0:
            os.rmdir(performance_dir)
        if len(os.listdir(cnn_dir)) == 0:
            os.rmdir(cnn_dir)

    prediction_df.reset_index(drop=True, inplace=True)
    metrics_df.reset_index(drop=True, inplace=True)
    mode_level_to_tsvs(output_dir, prediction_df, metrics_df, fold, selection, mode, dataset)


def retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, num_cnn):
    """Retrieve performance_df for single or multi-CNN framework.
    If the results of the multi-CNN were not concatenated it will be done here."""
    result_tsv = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection,
                              '%s_%s_level_prediction.tsv' % (dataset, mode))
    if os.path.exists(result_tsv):
        performance_df = pd.read_csv(result_tsv, sep='\t')

    else:
        concat_multi_cnn_results(output_dir, fold, selection, mode, dataset, num_cnn)
        performance_df = pd.read_csv(result_tsv, sep='\t')

    return performance_df


def soft_voting_to_tsvs(output_dir, fold, selection, mode, dataset='test', num_cnn=None, selection_threshold=None):
    """
    Writes soft voting results in tsv files.

    Args:
        output_dir: (str) path to the output directory.
        fold: (int) Fold number of the cross-validation.
        selection: (str) criterion on which the model is selected (either best_loss or best_acc)
        mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
        dataset: (str) name of the dataset for which the soft-voting is performed. If different from training or
            validation, the weights of soft voting will be computed on validation accuracies.
        num_cnn: (int) if given load the patch level results of a multi-CNN framework.
        selection_threshold: (float) all patches for which the classification accuracy is below the
            threshold is removed.
    """

    # Choose which dataset is used to compute the weights of soft voting.
    if dataset in ['train', 'validation']:
        validation_dataset = dataset
    else:
        validation_dataset = 'validation'
    test_df = retrieve_sub_level_results(output_dir, fold, selection, mode, dataset, num_cnn)
    validation_df = retrieve_sub_level_results(output_dir, fold, selection, mode, validation_dataset, num_cnn)

    performance_path = os.path.join(output_dir, 'fold-%i' % fold, 'cnn_classification', selection)
    if not os.path.exists(performance_path):
        os.makedirs(performance_path)

    df_final, metrics = soft_voting(test_df, validation_df, mode, selection_threshold=selection_threshold)

    df_final.to_csv(os.path.join(os.path.join(performance_path, '%s_image_level_prediction.tsv' % dataset)),
                    index=False, sep='\t')

    pd.DataFrame(metrics, index=[0]).to_csv(os.path.join(performance_path, '%s_image_level_metrics.tsv' % dataset),
                                            index=False, sep='\t')


def soft_voting(performance_df, validation_df, mode, selection_threshold=None):
    """
    Computes soft voting based on the probabilities in performance_df. Weights are computed based on the accuracies
    of validation_df.

    ref: S. Raschka. Python Machine Learning., 2015

    Args:
        performance_df: (DataFrame) results on patch level of the set on which the combination is made.
        validation_df: (DataFrame) results on patch level of the set used to compute the weights.
        mode: (str) input used by the network. Chosen from ['patch', 'roi', 'slice'].
        selection_threshold: (float) if given, all patches for which the classification accuracy is below the
            threshold is removed.

    Returns:
        df_final (DataFrame) the results on the image level
        results (dict) the metrics on the image level
    """

    # Compute the sub-level accuracies on the validation set:
    validation_df["accurate_prediction"] = validation_df.apply(lambda x: check_prediction(x), axis=1)
    sub_level_accuracies = validation_df.groupby("%s_id" % mode)["accurate_prediction"].sum()
    if selection_threshold is not None:
        sub_level_accuracies[sub_level_accuracies < selection_threshold] = 0
    weight_series = sub_level_accuracies / sub_level_accuracies.sum()

    # Sort to allow weighted average computation
    performance_df.sort_values(['participant_id', 'session_id', '%s_id' % mode], inplace=True)
    weight_series.sort_index(inplace=True)

    # Soft majority vote
    columns = ['participant_id', 'session_id', 'true_label', 'predicted_label']
    df_final = pd.DataFrame(columns=columns)
    for (subject, session), subject_df in performance_df.groupby(['participant_id', 'session_id']):
        y = subject_df["true_label"].unique().item()
        proba0 = np.average(subject_df["proba0"], weights=weight_series)
        proba1 = np.average(subject_df["proba1"], weights=weight_series)
        proba_list = [proba0, proba1]
        y_hat = proba_list.index(max(proba_list))

        row = [[subject, session, y, y_hat]]
        row_df = pd.DataFrame(row, columns=columns)
        df_final = df_final.append(row_df)

    results = evaluate_prediction(df_final.true_label.values.astype(int),
                                  df_final.predicted_label.values.astype(int))

    return df_final, results


def check_prediction(row):
    if row["true_label"] == row["predicted_label"]:
        return 1
    else:
        return 0

##################
# Multi - Task - #
##################

def train_multitask(model, train_loader, valid_loader, criterion, optimizer, resume, log_dir, model_dir, options):
    """
    Function used to train a CNN.
    The best model and checkpoint will be found in the 'best_model_dir' of options.output_dir.

    Args:
        model: (Module) CNN to be trained
        train_loader: (DataLoader) wrapper of the training dataset
        valid_loader: (DataLoader) wrapper of the validation dataset
        criterion: (loss) function to calculate the loss
        optimizer: (torch.optim) optimizer linked to model parameters
        resume: (bool) if True, a begun job is resumed
        log_dir: (str) path to the folder containing the logs
        model_dir: (str) path to the folder containing the models weights and biases
        options: (Namespace) ensemble of other options given to the main script.
    """
    print('multiclass training is: ' + str(options.multiclass))

    from tensorboardX import SummaryWriter
    from time import time

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(log_dir)

    # Create writers

    # Create tsv
    if options.num_labels == 4:
        columns = ['epoch', 'iteration', 'bacc_train_1', 'bacc_train_2', 'bacc_train_3', 'bacc_train_4',
               'mean_loss_train_1', 'mean_loss_train_2', 'mean_loss_train_3', 'mean_loss_train_4', 'mean_loss_train',
                'bacc_valid_1', 'bacc_valid_2', 'bacc_valid_3', 'bacc_valid_4',
               'mean_loss_valid_1', 'mean_loss_valid_2', 'mean_loss_valid_3', 'mean_loss_valid_4', 'mean_loss_valid']
    elif options.num_labels == 3:
        columns = ['epoch', 'iteration', 'bacc_train_1', 'bacc_train_2', 'bacc_train_3',
               'mean_loss_train_1', 'mean_loss_train_2', 'mean_loss_train_3', 'mean_loss_train',
                   'bacc_valid_1', 'bacc_valid_2', 'bacc_valid_3',
               'mean_loss_valid_1', 'mean_loss_valid_2', 'mean_loss_valid_3', 'mean_loss_valid']
    elif options.num_labels == 2:
        columns = ['epoch', 'iteration', 'bacc_train_1', 'bacc_train_2',
               'mean_loss_train_1', 'mean_loss_train_2', 'mean_loss_train' ,
                   'bacc_valid_1', 'bacc_valid_2',
               'mean_loss_valid_1', 'mean_loss_valid_2', 'mean_loss_valid']


    filename = os.path.join(log_dir, 'training.tsv')

    ## read criterion in case of 2 criterion

    criterion_1 = criterion[0]
    criterion_2 = criterion[1]

    # Initialize variables
    best_valid_accuracy = 0.0
    best_valid_loss = np.inf
    epoch = options.beginning_epoch



    model.train()  # set the module to training mode

    early_stopping = EarlyStopping('min', min_delta=options.tolerance, patience=options.patience)
    mean_loss_valid = None

    while epoch < options.epochs and not early_stopping.step(mean_loss_valid):
        print("At %d-th epoch." % epoch)

        model.zero_grad()
        evaluation_flag = True
        step_flag = True
        tend = time()
        total_time = 0

        for i, data in enumerate(train_loader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if options.gpu:
                if options.num_labels == 4:
                    imgs, labels1, labels2, labels3, labels4 = data['image'].cuda(), data['label_1'].cuda(), \
                                                           data['label_2'].cuda(), data['label_3'].cuda(), data['label_4'].cuda()
                elif options.num_labels == 3:
                    imgs, labels1, labels2, labels3 = data['image'].cuda(), data['label_1'].cuda(), \
                                                               data['label_2'].cuda(), data['label_3'].cuda()
                elif options.num_labels == 2:
                    imgs, labels1, labels2 = data['image'].cuda(), data['label_1'].cuda(), \
                                                               data['label_2'].cuda()
            else:
                imgs, labels1, labels2, labels3, labels4 = data['image'], data['label_1'], data['label_2'], data['label_3'], data['label_4']

            ### to delete after test
            imgs[imgs != imgs] = 0
            imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

            if options.num_labels == 4:
                train_output_1, train_output_2, train_output_3, train_output_4 = model(imgs)
            elif options.num_labels == 3:
                train_output_1, train_output_2, train_output_3 = model(imgs)
            elif options.num_labels == 2:
                train_output_1, train_output_2 = model(imgs)

            #_, predict_batch = train_output.topk(1) ### WHERE DO I USE IT ?

            loss_1 = criterion_1(train_output_1, labels1)
            loss_2 = criterion_2(train_output_2, labels2)

            if options.num_labels == 4:
                loss_3 = criterion_2(train_output_3, labels3)
                loss_4 = criterion_2(train_output_4, labels4)
                loss = loss_1 + loss_2 + loss_3 + loss_4
            elif options.num_labels == 3:
                loss_3 = criterion_2(train_output_3, labels3)
                loss = loss_1 + loss_2 + loss_3
            elif options.num_labels == 2:
                loss = loss_1 + loss_2

            # Back propagation
            loss.backward()

            del imgs, labels1, labels2#, labels3, labels4

            if (i + 1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                del loss

                # Evaluate the model only when no gradients are accumulated
                if options.evaluation_steps != 0 and (i + 1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    print('Iteration %d' % i)

                    if options.multiclass == False:
                        _, results_train = test_multitask(model, train_loader, options.gpu, criterion, multiclass=False, num_labels=options.num_labels)
                        _, results_valid = test_multitask(model, valid_loader, options.gpu, criterion, multiclass=False, num_labels=options.num_labels)
                    elif options.multiclass == True:
                        _, results_train = test_multitask(model, train_loader, options.gpu, criterion, multiclass=True, num_labels=options.num_labels)
                        _, results_valid = test_multitask(model, valid_loader, options.gpu, criterion, multiclass=True, num_labels=options.num_labels)


                    mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)
                    mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
                    model.train()

                    global_step = i + epoch * len(train_loader)
                    d_train = (len(train_loader) * train_loader.batch_size)
                    d_valid = (len(valid_loader) * valid_loader.batch_size)
                    # Write results on the dataframe
                    if options.num_labels == 4:
                        row = np.array([epoch, i, results_train["balanced_accuracy_1"],
                                        results_train["balanced_accuracy_2"], results_train["balanced_accuracy_3"],
                                        results_train["balanced_accuracy_4"],
                                        results_train["total_loss_1"] / d_train, results_train["total_loss_2"] / d_train,
                                        results_train["total_loss_3"] / d_train, results_train["total_loss_4"] / d_train,
                                        mean_loss_train,
                                        results_valid["balanced_accuracy_1"],
                                        results_valid["balanced_accuracy_2"],results_valid["balanced_accuracy_3"],
                                        results_valid["balanced_accuracy_4"],
                                        results_valid["total_loss_1"] / d_valid, results_valid["total_loss_2"] / d_valid,
                                        results_valid["total_loss_3"] / d_valid, results_valid["total_loss_4"] / d_valid,
                                        mean_loss_valid]).reshape(1, -1)
                    elif options.num_labels == 3:
                        row = np.array([epoch, i, results_train["balanced_accuracy_1"],
                                        results_train["balanced_accuracy_2"], results_train["balanced_accuracy_3"],
                                        results_train["total_loss_1"] / d_train, results_train["total_loss_2"] / d_train,
                                        results_train["total_loss_3"] / d_train,
                                        mean_loss_train,
                                        results_valid["balanced_accuracy_1"],
                                        results_valid["balanced_accuracy_2"],results_valid["balanced_accuracy_3"],
                                        results_valid["total_loss_1"] / d_valid, results_valid["total_loss_2"] / d_valid,
                                        results_valid["total_loss_3"] / d_valid,
                                        mean_loss_valid]).reshape(1, -1)
                    elif options.num_labels == 2:
                        row = np.array([epoch, i, results_train["balanced_accuracy_1"],
                                        results_train["balanced_accuracy_2"],
                                        results_train["tota_loss_1"] / d_train, results_train["total_loss_2"] / d_train,
                                        mean_loss_train,
                                        results_valid["balanced_accuracy_1"],
                                        results_valid["balanced_accuracy_2"],
                                        results_valid["tota_loss_1"] / d_valid, results_valid["total_loss_2"] / d_valid,
                                        mean_loss_valid]).reshape(1, -1)


                    row_df = pd.DataFrame(row, columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')

                    print("%s level training accuracy is %f at the end of iteration %d"
                          % (options.mode, results_train["balanced_accuracy"], i))
                    print("%s level validation accuracy is %f at the end of iteration %d"
                          % (options.mode, results_valid["balanced_accuracy"], i))

            tend = time()
        print('Mean time per batch loading (train):', total_time / len(train_loader) * train_loader.batch_size)

        # If no step has been performed, raise Exception
        if step_flag:
            raise Exception('The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        elif evaluation_flag and options.evaluation_steps != 0:
            warnings.warn('Your evaluation steps are too big compared to the size of the dataset.'
                          'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        model.zero_grad()
        print('Last checkpoint at the end of the epoch %d' % epoch)

        if options.multiclass == False:
            _, results_train = test_multitask(model, train_loader, options.gpu, criterion, multiclass=False, num_labels=options.num_labels)
            _, results_valid = test_multitask(model, valid_loader, options.gpu, criterion, multiclass=False, num_labels=options.num_labels)
        elif options.multiclass == True:
            _, results_train = test_multitask(model, train_loader, options.gpu, criterion, multiclass=True, num_labels=options.num_labels)
            _, results_valid = test_multitask(model, valid_loader, options.gpu, criterion, multiclass=True, num_labels=options.num_labels)


        mean_loss_train = results_train["total_loss"] / (len(train_loader) * train_loader.batch_size)
        mean_loss_valid = results_valid["total_loss"] / (len(valid_loader) * valid_loader.batch_size)
        model.train()
        d_train = (len(train_loader) * train_loader.batch_size)
        d_valid = (len(valid_loader) * valid_loader.batch_size)
        global_step = (epoch + 1) * len(train_loader)

        # Write results on the dataframe
        if options.num_labels == 4:
            row = np.array([epoch, i, results_train["balanced_accuracy_1"],
                            results_train["balanced_accuracy_2"], results_train["balanced_accuracy_3"],
                            results_train["balanced_accuracy_4"],
                           results_train["total_loss_1"] / d_train, results_train["total_loss_2"] / d_train,
                           results_train["total_loss_3"] / d_train, results_train["total_loss_4"] / d_train,
                            mean_loss_train,
                            results_valid["balanced_accuracy_1"],
                            results_valid["balanced_accuracy_2"], results_valid["balanced_accuracy_3"],
                            results_valid["balanced_accuracy_4"],
                           results_valid["total_loss_1"] / d_valid, results_valid["total_loss_2"] / d_valid,
                           results_valid["total_loss_3"] / d_valid, results_valid["total_loss_4"] / d_valid,
                           mean_loss_valid]).reshape(1, -1)
        elif options.num_labels == 3:
            row = np.array([epoch, i, results_train["balanced_accuracy_1"],
                            results_train["balanced_accuracy_2"], results_train["balanced_accuracy_3"],
                            results_train["total_loss_1"] / d_train, results_train["total_loss_2"] / d_train,
                            results_train["total_loss_3"] / d_train,
                            mean_loss_train, results_valid["balanced_accuracy_1"],
                            results_valid["balanced_accuracy_2"], results_valid["balanced_accuracy_3"],
                            results_valid["total_loss_1"] / d_valid, results_valid["total_loss_2"] / d_valid,
                            results_valid["total_loss_3"] / d_valid,
                            mean_loss_valid]).reshape(1, -1)
        elif options.num_labels == 2:
            row = np.array([epoch, i, results_train["balanced_accuracy_1"],
                            results_train["balanced_accuracy_2"],
                            results_train["total_loss_1"] / d_train, results_train["total_loss_2"] / d_train,
                            mean_loss_train,
                            results_valid["balanced_accuracy_1"], results_valid["balanced_accuracy_2"],
                            results_valid["total_loss_1"] / d_valid, results_valid["total_loss_2"] / d_valid,
                            mean_loss_valid]
                           ).reshape(1, -1)

        row_df = pd.DataFrame(row, columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')

        if options.num_labels == 4:
            average_balanced_accuracy = (results_valid["balanced_accuracy_1"].item() +
                                         results_valid["balanced_accuracy_2"].item() +
                                        results_valid["balanced_accuracy_3"].item() +
                                         results_valid["balanced_accuracy_4"].item())/4
            average_accuracy = (results_valid["accuracy_1"].item() +
                                results_valid["accuracy_2"].item() +
                                results_valid["accuracy_3"].item() +
                                results_valid["accuracy_4"].item())/4
        elif options.num_labels == 3:
            average_balanced_accuracy = (results_valid["balanced_accuracy_1"].item() +
                                         results_valid["balanced_accuracy_2"].item() +
                                         results_valid["balanced_accuracy_3"].item()) / 3
            average_accuracy = (results_valid["accuracy_1"].item() +
                                results_valid["accuracy_2"].item() +
                                results_valid["accuracy_3"].item()) / 3
        elif options.num_labels == 2:
            average_balanced_accuracy = (results_valid["balanced_accuracy_1"].item() +
                                         results_valid["balanced_accuracy_2"].item()) / 2
            average_accuracy = (results_valid["accuracy_1"].item() +
                                results_valid["accuracy_2"].item()) / 2

        print("%s level training accuracy is %f at the end of iteration %d"
              % (options.mode, average_balanced_accuracy, len(train_loader)))
        print("%s level validation accuracy is %f at the end of iteration %d"
              % (options.mode, average_accuracy, len(train_loader)))

        accuracy_is_best = average_balanced_accuracy > best_valid_accuracy
        loss_is_best = mean_loss_valid < best_valid_loss
        best_valid_accuracy = max(average_accuracy, best_valid_accuracy)
        best_valid_loss = min(mean_loss_valid, best_valid_loss)

        save_checkpoint({'model': model.state_dict(),
                         'epoch': epoch,
                         'valid_loss': mean_loss_valid,
                         'valid_acc': average_balanced_accuracy},
                        accuracy_is_best, loss_is_best,
                        model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': options.optimizer,
                         },
                        False, False,
                        model_dir,
                        filename='optimizer.pth.tar')

        epoch += 1

    os.remove(os.path.join(model_dir, "optimizer.pth.tar"))
    os.remove(os.path.join(model_dir, "checkpoint.pth.tar"))


def test_multitask(model, dataloader, use_cuda, criterion, mode="image", multiclass=False, num_labels=2,
                   classify_function=False):
    """
    Computes the predictions and evaluation metrics.

    Args:
        model: (Module) CNN to be tested.
        dataloader: (DataLoader) wrapper of a dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.
        mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
    Returns
        (DataFrame) results of each input.
        (dict) ensemble of metrics + total loss on mode level.
    """
    print('thesea are the number of labels is')
    print(num_labels)
    model.eval()

    if mode == "image":
        if num_labels == 4:
            columns = ["participant_id", "session_id", "true_label_1", "predicted_label_1",
                       "true_label_2", "predicted_label_2", "true_label_3", "predicted_label_3",
                       "true_label_4", "predicted_label_4"]
        elif num_labels == 3:
            columns = ["participant_id", "session_id", "true_label_1", "predicted_label_1",
                       "true_label_2", "predicted_label_2", "true_label_3", "predicted_label_3"]
        elif num_labels == 2:
            columns = ["participant_id", "session_id", "true_label_1", "predicted_label_1",
                       "true_label_2", "predicted_label_2"]

    elif mode in ["patch", "roi", "slice"]:
        columns = ['participant_id', 'session_id', '%s_id' % mode, 'true_label', 'predicted_label', 'proba0', 'proba1']
    else:
        raise ValueError("The mode %s is invalid." % mode)

    if classify_function == False:
    ## read 2 criterions if i am in the training
        criterion_1 = criterion[0]
        criterion_2 = criterion[1]
    else:
        criterion = criterion

    softmax = torch.nn.Softmax(dim=1)
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0
    total_loss_1 = 0
    total_loss_2 = 0
    total_loss_3 = 0
    total_loss_4 = 0
    total_time = 0
    tend = time()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            t0 = time()
            total_time = total_time + t0 - tend
            if use_cuda:
                if num_labels == 4:
                    inputs, labels1, labels2, labels3, labels4 = data['image'].cuda(), data['label_1'].cuda(), \
                                                           data['label_2'].cuda(), data['label_3'].cuda(), data[
                                                               'label_4'].cuda()
                elif num_labels == 3:
                    inputs, labels1, labels2, labels3 = data['image'].cuda(), data['label_1'].cuda(), \
                                                                 data['label_2'].cuda(), data['label_3'].cuda()
                elif num_labels == 2:
                    inputs, labels1, labels2 = data['image'].cuda(), data['label_1'].cuda(), \
                                                                 data['label_2'].cuda()

            else:
                inputs, labels1, labels2, labels3, labels4 = data['image'], data['label_1'], data['label_2'], \
                                                             data['label_3'], data['label_4']


            inputs[inputs != inputs] = 0
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())

            if num_labels == 4:
                outputs1, outputs2, outputs3, outputs4 = model(inputs)
                if classify_function == False:
                    loss1 = criterion_1(outputs1, labels1)
                    loss2 = criterion_2(outputs2, labels2)
                    loss3 = criterion_2(outputs3, labels3)
                    loss4 = criterion_2(outputs4, labels4)
                else:
                    loss1 = criterion(outputs1, labels1)
                    loss2 = criterion(outputs2, labels2)
                    loss3 = criterion(outputs3, labels3)
                    loss4 = criterion(outputs4, labels4)
                total_loss_1 += loss1.item()
                total_loss_2 += loss2.item()
                total_loss_3 += loss3.item()
                total_loss_4 += loss4.item()
                loss = loss1 + loss2 + loss3 + loss4
            elif num_labels == 3:
                outputs1, outputs2, outputs3 = model(inputs)
                if classify_function == False:
                    loss1 = criterion_1(outputs1, labels1)
                    loss2 = criterion_2(outputs2, labels2)
                    loss3 = criterion_2(outputs3, labels3)
                else:
                    loss1 = criterion(outputs1, labels1)
                    loss2 = criterion(outputs2, labels2)
                    loss3 = criterion(outputs3, labels3)
                total_loss_1 += loss1.item()
                total_loss_2 += loss2.item()
                total_loss_3 += loss3.item()
                loss = loss1 + loss2 + loss3
            elif num_labels == 2:
                outputs1, outputs2 = model(inputs)
                if classify_function == False:
                    loss1 = criterion_1(outputs1, labels1)
                    loss2 = criterion_2(outputs2, labels2)
                else:
                    loss1 = criterion(outputs1, labels1)
                    loss2 = criterion(outputs2, labels2)
                total_loss_1 += loss1.item()
                total_loss_2 += loss2.item()
                loss = loss1 + loss2


            total_loss += loss.item()
            _, predicted_1 = torch.max(outputs1.data, 1)
            _, predicted_2 = torch.max(outputs2.data, 1)
            if num_labels == 3:
                _, predicted_3 = torch.max(outputs3.data, 1)
            elif num_labels == 4:
                _, predicted_3 = torch.max(outputs3.data, 1)
                _, predicted_4 = torch.max(outputs4.data, 1)

            # Generate detailed DataFrame
            for idx, sub in enumerate(data['participant_id']):
                if mode == "image":
                    if num_labels == 4:
                        row = [[sub, data['session_id'][idx], labels1[idx].item(), predicted_1[idx].item(),
                            labels2[idx].item(), predicted_2[idx].item(), labels3[idx].item(), predicted_3[idx].item(),
                            labels4[idx].item(), predicted_4[idx].item()]]
                    elif num_labels == 3:
                        row = [[sub, data['session_id'][idx], labels1[idx].item(), predicted_1[idx].item(),
                            labels2[idx].item(), predicted_2[idx].item(), labels3[idx].item(), predicted_3[idx].item()]]
                    elif num_labels == 2:
                        row = [[sub, data['session_id'][idx], labels1[idx].item(), predicted_1[idx].item(),
                            labels2[idx].item(), predicted_2[idx].item()]]


                else: # not for multi-task since we are only working on images
                    normalized_output = softmax(outputs)
                    row = [[sub, data['session_id'][idx], data['%s_id' % mode][idx].item(),
                            labels[idx].item(), predicted[idx].item(),
                            normalized_output[idx, 0].item(), normalized_output[idx, 1].item()]]

                row_df = pd.DataFrame(row, columns=columns)
                results_df = pd.concat([results_df, row_df])

            del inputs, outputs2, loss
            tend = time()
        print('Mean time per batch loading (test):', total_time / len(dataloader) * dataloader.batch_size)
        results_df.reset_index(inplace=True, drop=True)

        # calculate the balanced accuracy
        if multiclass == False:
            results_1 = evaluate_prediction(results_df.true_label_1.values.astype(int),
                                      results_df.predicted_label_1.values.astype(int))
            results_2 = evaluate_prediction(results_df.true_label_2.values.astype(int),
                                            results_df.predicted_label_2.values.astype(int))
            if num_labels == 3:
                results_3 = evaluate_prediction(results_df.true_label_3.values.astype(int),
                                            results_df.predicted_label_3.values.astype(int))
            elif num_labels == 4:
                results_3 = evaluate_prediction(results_df.true_label_3.values.astype(int),
                                                results_df.predicted_label_3.values.astype(int))
                results_4 = evaluate_prediction(results_df.true_label_4.values.astype(int),
                                            results_df.predicted_label_4.values.astype(int))

        elif multiclass == True:
            results_1 = evaluate_prediction_multiclass(results_df.true_label_1.values.astype(int),
                                      results_df.predicted_label_1.values.astype(int))
            results_2 = evaluate_prediction_multiclass(results_df.true_label_2.values.astype(int),
                                                     results_df.predicted_label_2.values.astype(int))
            if num_labels == 3:
                results_3 = evaluate_prediction_multiclass(results_df.true_label_3.values.astype(int),
                                                     results_df.predicted_label_3.values.astype(int))
            elif num_labels == 4:
                results_4 = evaluate_prediction_multiclass(results_df.true_label_4.values.astype(int),
                                                     results_df.predicted_label_4.values.astype(int))



        results_df.reset_index(inplace=True, drop=True)

        if num_labels == 4:
            results ={'accuracy_1': results_1['accuracy'], 'balanced_accuracy_1': results_1['balanced_accuracy'],
                    'accuracy_2': results_2['accuracy'], 'balanced_accuracy_2': results_2['balanced_accuracy'],
                     'accuracy_3': results_3['accuracy'], 'balanced_accuracy_3': results_3['balanced_accuracy'],
                     'accuracy_4': results_4['accuracy'], 'balanced_accuracy_4': results_4['balanced_accuracy'],
                      'total_loss_1': total_loss_1, 'total_loss_2': total_loss_2, 'total_loss_3': total_loss_3,
                      'total_loss_4': total_loss_4}
        elif num_labels == 3:
            results ={'accuracy_1': results_1['accuracy'], 'balanced_accuracy_1': results_1['balanced_accuracy'],
                    'accuracy_2': results_2['accuracy'], 'balanced_accuracy_2': results_2['balanced_accuracy'],
                     'accuracy_3': results_3['accuracy'], 'balanced_accuracy_3': results_3['balanced_accuracy'],
                      'total_loss_1': total_loss_1, 'total_loss_2': total_loss_2, 'total_loss_3': total_loss_3}
        elif num_labels == 2:
            results = {'accuracy_1': results_1['accuracy'], 'balanced_accuracy_1': results_1['balanced_accuracy'],
                       'accuracy_2': results_2['accuracy'], 'balanced_accuracy_2': results_2['balanced_accuracy'],
                      'total_loss_1': total_loss_1, 'total_loss_2': total_loss_2}

        results['total_loss'] = total_loss
        torch.cuda.empty_cache()



    return results_df, results
