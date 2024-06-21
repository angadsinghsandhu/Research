import torch

# Function to calculate R-squared accuracy
def calculate_r_squared_accuracy(predictions, targets):
    # Check input validity
    assert predictions.shape == targets.shape, "Predictions and targets should have the same shape."
    assert not torch.isnan(predictions).any() and not torch.isnan(targets).any(), "Input contains NaN values."
    assert not torch.isinf(predictions).any() and not torch.isinf(targets).any(), "Input contains Inf values."

    # Convert to float if necessary
    predictions = predictions.float()
    targets = targets.float()

    # Calculate residual sum of squares (SS Residual)
    ss_res = torch.sum((targets - predictions) ** 2)

    # Calculate total sum of squares (SS Total) with respect to the mean of targets
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)

    # Ensure ss_tot is not zero to avoid division by zero
    if ss_tot == 0:
        raise ValueError("Total sum of squares is zero; all target values are identical.")

    # Calculate R-squared
    r_squared = 1 - ss_res / ss_tot

    # Calculate accuracy based on a baseline model (predicting the mean)
    baseline_predictions = torch.mean(targets).repeat(len(targets))
    ss_res_baseline = torch.sum((targets - baseline_predictions) ** 2)
    r_squared_baseline = 1 - ss_res_baseline / ss_tot

    # Calculate accuracy improvement over the baseline model
    r_squared_accuracy = max(0, (r_squared / r_squared_baseline) * 100)

    return r_squared.item(), r_squared_accuracy

# Function to calculate MSE accuracy
def calculate_mse_accuracy(predictions, targets):
    # Check input validity
    assert predictions.shape == targets.shape, "Predictions and targets should have the same shape."
    assert not torch.isnan(predictions).any() and not torch.isnan(targets).any(), "Input contains NaN values."
    assert not torch.isinf(predictions).any() and not torch.isinf(targets).any(), "Input contains Inf values."

    # Convert to float if necessary
    predictions = predictions.float()
    targets = targets.float()

    # Calculate the Mean Squared Error
    mse = torch.mean((targets - predictions) ** 2)

    # An example way to convert MSE to an accuracy measure
    # If zero MSE gives 100% accuracy, calculate a decreasing proportion
    mse_accuracy = max(0, 100 - mse.item())

    return mse.item(), mse_accuracy

# Function to calculate accuracy
def calculate_accuracy(predictions, targets, threshold=0.1):
    # Calculate the absolute difference
    abs_difference = torch.abs(predictions - targets)
    # Check which differences are below the threshold
    accurate_predictions = abs_difference <= threshold
    # Calculate the accuracy as the percentage of correct predictions
    accuracy = accurate_predictions.sum().item() / targets.numel()
    return accuracy

## EARLY STOPPING SETUP
class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print("Initial validation loss recorded.")
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
                return True
        else:
            if self.verbose and val_loss < self.best_loss:
                print("Validation loss improved.")
            self.best_loss = val_loss
            self.counter = 0
        return False

    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
