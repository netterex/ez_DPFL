import copy


def create_clients(net, num_clients):
    clients_model_list = []
    for i in range(num_clients):
        clients_model_list.append(copy.deepcopy(net))

    return clients_model_list
