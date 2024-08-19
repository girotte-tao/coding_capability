import torch
import torch.nn as nn

# Example: Assume we have 3 classes
logits = torch.tensor([[1.0, 2.0, 0.5]])  # Model output (before softmax)
labels = torch.tensor([1])  # True label index

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

# Compute the loss
loss = loss_fn(logits, labels)
print(loss.item())  # Outputs the scalar loss value


# 1. softmax, get a probability distribution
# 2. get the log probability of the true class e_i/sum(e_j)
# 3. get the negative log probability of the true class -log(e_i/sum(e_j))
# softmax is not idempotent
# the larger the cross entropy, the larger the difference between the true class and the predicted class