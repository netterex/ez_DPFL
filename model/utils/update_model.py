import pandas as pd
import torch

from model.utils.noise import add_noise_on_param
from model.utils.validation import validation


def send_center_model_to_clients(center_model, clients_model_list):
    with torch.no_grad():
        for client_model in clients_model_list:
            client_model.load_state_dict(center_model.state_dict(), strict=True)


def aggregator(center_model, total_grads, clients_data_list, lr):
    number_of_data_on_each_clients = [len(clients_data_list[i]) for i in range(len(clients_data_list))]
    total_data_length = sum(number_of_data_on_each_clients)
    weight_of_each_clients = [x / total_data_length for x in number_of_data_on_each_clients]

    # 初始化全局梯度为零
    global_grads = {name: torch.zeros_like(param) for name, param in center_model.named_parameters()}

    # 对每个客户端的梯度进行加权平均并累加到全局梯度中
    for client_grads, weight in zip(total_grads, weight_of_each_clients):
        for name, grad in client_grads.items():
            if name in global_grads:
                global_grads[name] += grad * weight

    # 使用累加的全局梯度来更新全局模型的参数
    with torch.no_grad():
        for param_name, param in center_model.named_parameters():
            if param_name in global_grads:
                param -= lr * global_grads[param_name]
