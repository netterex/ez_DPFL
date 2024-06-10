from model.CNN import CNNMnist
from model.CNN import CNNFmnist
from model.CNN import CNNCifar


def get_model(model, dataset, device):
    if model == 'CNN' and dataset == 'MNIST':
        net = CNNMnist().to(device)
    elif model == 'CNN' and dataset == 'FMNIST':
        net = CNNFmnist().to(device)
    elif model == 'CNN' and dataset == 'CIFAR-10':
        net = CNNCifar().to(device)
    else:
        raise ValueError("Unsupported dataset name.")

    return net
