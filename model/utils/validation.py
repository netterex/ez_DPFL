import torch
import torch.nn.functional as F


def validation(model, test_loader, device):
    model.eval()
    num_samples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum')
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_samples += len(data)

    test_loss /= num_samples
    test_accuracy = correct / num_samples

    return test_loss, test_accuracy
