import torch


def train_step(model, inputs, targets, loss_fn, optimizer, scheduler):
    model.train()
    # forward pass
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)
    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return {"loss": loss}


def evaluate_step(model, inputs, targets, loss_fn):
    model.eval()
    # forward pass
    with torch.no_grad():
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
    return {"loss": loss, "targets": targets, "predictions": predictions}
