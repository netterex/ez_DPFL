import matplotlib.pyplot as plt


def draw_picture(data_dir, test_acc_list):
    # 绘制全局模型准确率的变化图
    epochs = list(range(1, len(test_acc_list) + 1))
    plt.plot(epochs, test_acc_list)
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'{data_dir}/model_acc.svg', format='svg')
    plt.show()
