import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from model.utils.noise import add_noise_on_grads


def train_one_epoch(model, train_dataloader, device, data_size, clip, epsilon, lr, momentum):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_loss = 0
    total = 0
    correct = 0
    noisy_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for data, target in train_dataloader:
        data, target = data.to(device), target.to(device)
        total += data.size(0)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 累加带噪声的梯度
                    noisy_grads[name] += param.grad.clone()

        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        correct += (predicted == target).sum().item()

        optimizer.step()

    # 调用函数进行加噪
    add_noise_on_grads(model, data_size, clip, device, epsilon, noisy_grads)

    # 计算平均损失和准确率
    train_loss /= total
    train_accuracy = correct / total

    # 返回平均损失、准确率和添加噪声后的平均梯度
    return train_loss, train_accuracy, noisy_grads


def train_with_dp(clients_data_list, clients_model_list, batch_size_of_each_client, num_clients, epochs, device,
                  clip, epsilon, lr, momentum):
    total_grads = []

    for i in range(num_clients):
        # math.floor 向下取整
        data_size = len(clients_data_list[i])
        batch_size = batch_size_of_each_client[i]
        train_dataloader = DataLoader(clients_data_list[i], batch_size=batch_size, shuffle=False,
                                      drop_last=True)
        model = clients_model_list[i]

        noisy_grads = None

        for epoch in range(epochs):
            # 客户端本地具体训练过程，得到更新后的梯度
            train_loss, train_accuracy, noisy_grads = train_one_epoch(model, train_dataloader, device,
                                                                                      data_size, clip, epsilon, lr,
                                                                                      momentum)
            print("----------------------------------------------------------------------------")
            print(
                f"Client {i + 1}, Epoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Train Accuracy: {100 * train_accuracy:.4f}%")

        total_grads.append(noisy_grads)

    return total_grads
