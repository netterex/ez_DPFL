import os
import torch
import time

from data.utils.load_dataset import load_dataset
from data.utils.noniid_dirichlet import dataset_NonIID
from torch.utils.data import DataLoader
from model.utils.local_train import train_with_dp
from model.utils.update_model import send_center_model_to_clients, aggregator
from model.utils.validation import validation
from model.utils.get_model import get_model
from model.utils.create_client import create_clients
from utils.draw import draw_picture
from utils.load_config import load_config
from utils.log import log_results

if __name__ == "__main__":
    # 程序开始时间
    start_time = time.time()

    # 加载配置文件
    config_path = "config.yml"
    config = load_config(config_path)

    # 数据集、模型
    device = torch.device(config["device"])
    model = config["model"]
    dataset = config["dataset"]
    lr = config["learning_rate"]
    momentum = config["momentum"]

    # non-iid化（狄立克雷分布）
    alpha = config["alpha"]
    seed = int(config["seed"])
    q = config["q"]

    # 训练条件
    start_round = config["start_round"]
    iters = config["iters"]
    epochs = config["epochs"]
    num_clients = config["num_clients"]
    test_batch_size = config["test_batch_size"]
    current_time = config["current_time"]

    # 差分隐私
    clip = config["clip"]
    epsilon = config["epsilon"]

    # 获取当前时间
    if current_time == "" or current_time is None:
        current_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    # 创建保存目录
    dir_name = f"{current_time}_model_{model}_dataset_{dataset}_lr_{lr}_clients_{num_clients}_seed_{seed}_epsilon_{epsilon}"
    model_dir = os.path.join(f"./saved/{dir_name}", "model")
    data_dir = os.path.join(f"./saved/{dir_name}", "data")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # 加载数据集
    dataset_train, dataset_test = load_dataset(dataset)
    # 构建全局模型
    center_model = get_model(model, dataset, device)

    # 利用狄利克雷分布将数据non-iid化
    clients_data_list, weight_of_each_client, batch_size_of_each_client = dataset_NonIID(dataset_train, num_clients,
                                                                                         alpha, seed, q)
    # 为每个客户端创建本地模型
    clients_model_list = create_clients(center_model, num_clients)

    # 若从中间轮次开始训练，则加载参数到全局模型
    load_file = f"{model_dir}/model_optimizers_state_round_{start_round}.pt"
    if start_round > 0 and os.path.exists(load_file):
        # 加载模型
        state = torch.load(load_file)
        center_model.load_state_dict(state["model_state"])

    # 创建测试集加载器
    test_data_loader = DataLoader(dataset_test, batch_size=test_batch_size, shuffle=False)

    print("Start Federated Learning with Differential Privacy")
    # 存储每轮迭代的全局模型准确率和损失
    test_acc_list = []
    test_loss_list = []

    for i in range(iters):
        current_round = i + start_round + 1
        print(f"Iteration: {current_round}")
        # 中心服务器将全局模型分给各个客户端
        send_center_model_to_clients(center_model, clients_model_list)

        # 客户端训练本地模型，返回的是包含每个客户端训练后的梯度集合
        total_grads = train_with_dp(clients_data_list, clients_model_list,
                                                           batch_size_of_each_client, num_clients,
                                                           epochs, device, clip, epsilon, lr, momentum)

        # 全局模型聚合
        aggregator(center_model, total_grads, clients_data_list, lr)
        # 使用测试集对全局模型测试
        test_loss, test_acc = validation(center_model, test_data_loader, device)
        print("============================================================================")
        print(f"Iteration: {current_round}, Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_acc:.2f}%")
        print("============================================================================")
        test_acc_list.append(round(test_acc, 4))
        test_loss_list.append(test_loss)

        save_path = f"{model_dir}/model_optimizers_state_round_{start_round}.pt"
        state = {
            "model_state": center_model.state_dict()
        }
        torch.save(state, save_path)

    # 测试全局模型
    if test_loss_list is not None and test_acc_list is not None:
        draw_picture(data_dir, test_acc_list)
        log_results(start_round, test_loss_list, test_acc_list, data_dir)

    # 程序结束时间
    end_time = time.time()
    execution_time = int(end_time - start_time)
    print(f"程序执行时间：{execution_time // 3600} 小时 {execution_time // 60}分 {execution_time % 60}秒")
