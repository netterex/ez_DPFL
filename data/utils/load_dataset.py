from torchvision import datasets, transforms


def load_dataset(dataset):
    if dataset == 'MNIST':
        # MNIST数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset == 'CIFAR-10':
        # CIFAR-10数据集
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif dataset == 'FMNIST':
        # Fashion-MNIST数据集
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        dataset_train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        dataset_test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unsupported dataset name.")

    return dataset_train, dataset_test
