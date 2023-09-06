import torch
import pandas as pd
import numpy as np
import tqdm
import tqdm.notebook
import time
import logging

logger = logging.getLogger(__name__)

def train(
    model,
    loss_function,
    optimizer,
    training_dataloader,
    evaluation_dataloader,
    accuracy_function,
    num_epochs=5,
    device=None,
    progress_bar=False,
    notebook=False,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_diagnostics_list = list()
    evaluation_diagnostics_list = list()
    for epoch_index in range(num_epochs):
        logger.info(f'Starting training for epoch {epoch_index + 1} ({len(training_dataloader)} batches)')
        epoch_start_time = time.time()
        training_diagnostics_epoch = train_epoch(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            dataloader=training_dataloader,
            accuracy_function=accuracy_function,
            device=device,
            progress_bar=progress_bar,
            notebook=notebook,
        )
        epoch_end_time = time.time()
        epoch_time_elapsed = epoch_end_time - epoch_start_time
        num_training_examples = training_diagnostics_epoch['num_examples'].sum()
        logger.info(f'Processed {num_training_examples} training examples in {epoch_time_elapsed:.1f} seconds ({num_training_examples/epoch_time_elapsed:.1f} examples per second)')
        training_diagnostics_epoch = (
            training_diagnostics_epoch
            .reset_index()
            .assign(epoch_index=epoch_index)
            .set_index([
                'epoch_index',
                'batch_index'
            ])
        )
        training_diagnostics_list.append(training_diagnostics_epoch)
        logger.info(f'Starting evaluation for epoch {epoch_index + 1} ({len(evaluation_dataloader)} batches)')
        evaluation_start_time = time.time()
        evaluation_loss, evaluation_accuracy, num_evaluation_examples = evaluate(
            model=model,
            loss_function=loss_function,
            dataloader=evaluation_dataloader,
            accuracy_function=accuracy_function,
            device=device,
            progress_bar=progress_bar,
            notebook=notebook,
        )
        evaluation_end_time = time.time()
        evaluation_time_elapsed = evaluation_end_time - evaluation_start_time
        logger.info(f'Processed {num_evaluation_examples} evaluation examples in {evaluation_time_elapsed:.1f} seconds ({num_evaluation_examples/evaluation_time_elapsed:.1f} examples per second)')
        logger.info(f'Accuracy is {evaluation_accuracy}')
        evaluation_diagnostics_list.append({
            'epoch_index': epoch_index,
            'last_training_loss': training_diagnostics_epoch.iloc[-1]['loss'],
            'last_training_accuracy': training_diagnostics_epoch.iloc[-1]['accuracy'],
            'evaluation_loss': evaluation_loss,
            'evaluation_accuracy': evaluation_accuracy,
        })
    training_diagnostics = pd.concat(training_diagnostics_list)
    evaluation_diagnostics = (
        pd.DataFrame(evaluation_diagnostics_list)
        .set_index('epoch_index')
    )
    return training_diagnostics, evaluation_diagnostics

def train_epoch(
    model,
    loss_function,
    optimizer,
    dataloader,
    accuracy_function,
    device=None,
    progress_bar=False,
    notebook=False,
):
    epoch_start = time.time()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_batches = len(dataloader)
    num_examples_by_batch = np.zeros(num_batches, dtype=np.dtype(int))
    loss_by_batch = np.full(num_batches, np.nan)
    accuracy_by_batch = np.full(num_batches, np.nan)
    model.train()
    if progress_bar:
        if notebook:
            dataloader_iterator = tqdm.notebook.tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            dataloader_iterator = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
    else:
        dataloader_iterator = enumerate(dataloader)
    for batch_index, (x_batch, y_batch) in dataloader_iterator:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        model_output_batch = model(x_batch)
        loss_batch = loss_function(model_output_batch, y_batch)
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        with torch.no_grad():
            num_examples = len(x_batch)
            accuracy_batch = accuracy_function(model_output_batch, y_batch)
            num_examples_by_batch[batch_index] = num_examples
            loss_by_batch[batch_index] = loss_batch.item()
            accuracy_by_batch[batch_index] = accuracy_batch.item()
    epoch_end = time.time()
    epoch_time_elapsed = epoch_end - epoch_start

    diagnostics = (
        pd.DataFrame({
            'batch_index': range(num_batches),
            'num_examples': num_examples_by_batch,
            'loss': loss_by_batch,
            'accuracy': accuracy_by_batch
        })
        .set_index('batch_index')
    )
    return diagnostics

def evaluate(
    model,
    loss_function,
    dataloader,
    accuracy_function,
    device=None,
    progress_bar=False,
    notebook=False,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_batches = len(dataloader)
    num_examples_by_batch = np.zeros(num_batches, dtype=np.dtype(int))
    loss_by_batch = np.full(num_batches, np.nan)
    accuracy_by_batch = np.full(num_batches, np.nan)
    model.eval()
    with torch.no_grad():
        if progress_bar:
            if notebook:
                dataloader_iterator = tqdm.notebook.tqdm(enumerate(dataloader), total=len(dataloader))
            else:
                dataloader_iterator = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            dataloader_iterator = enumerate(dataloader)
        for batch_index, (x_batch, y_batch) in dataloader_iterator:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            num_examples = len(x_batch)
            model_output_batch = model(x_batch)
            loss_batch = loss_function(model_output_batch, y_batch)
            accuracy_batch = accuracy_function(model_output_batch, y_batch)
            num_examples_by_batch[batch_index] = num_examples
            loss_by_batch[batch_index] = loss_batch.item()
            accuracy_by_batch[batch_index] = accuracy_batch.item()
    loss = np.average(loss_by_batch, weights=num_examples_by_batch)
    accuracy = np.average(accuracy_by_batch, weights=num_examples_by_batch)
    num_examples = num_examples_by_batch.sum()
    return loss, accuracy, num_examples

def fraction_correct(model_output, y):
    num_examples = len(y)
    predictions = torch.argmax(model_output, dim=1)
    num_correct = (predictions == y).float().sum()
    accuracy = num_correct/num_examples
    return accuracy
