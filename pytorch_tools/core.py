import torch
import pandas as pd
import numpy as np
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
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_diagnostics_list = list()
    evaluation_diagnostics_list = list()
    for epoch_index in range(num_epochs):
        logger.info(f'Starting training for epoch {epoch_index}')
        training_diagnostics_epoch = train_epoch(
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            dataloader=training_dataloader,
            accuracy_function=accuracy_function,
            device=device,
        )
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
        logger.info(f'Starting validation for epoch {epoch_index}')
        evaluation_loss, evaluation_accuracy = evaluate(
            model=model,
            loss_function=loss_function,
            dataloader=evaluation_dataloader,
            accuracy_function=accuracy_function,
            device=device,
        )
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
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_batches = len(dataloader)
    num_examples_by_batch = np.full(num_batches, np.nan)
    loss_by_batch = np.full(num_batches, np.nan)
    accuracy_by_batch = np.full(num_batches, np.nan)
    model.train()
    for batch_index, (x_batch, y_batch) in enumerate(dataloader):
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
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    num_batches = len(dataloader)
    num_examples_by_batch = np.full(num_batches, np.nan)
    loss_by_batch = np.full(num_batches, np.nan)
    accuracy_by_batch = np.full(num_batches, np.nan)
    model.eval()
    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(dataloader):
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
    return loss, accuracy

def fraction_correct(model_output, y):
    num_examples = len(y)
    predictions = torch.argmax(model_output, dim=1)
    num_correct = (predictions == y).float().sum()
    accuracy = num_correct/num_examples
    return accuracy
