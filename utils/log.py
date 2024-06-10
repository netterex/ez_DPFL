import pandas as pd


def log_results(start_round, test_loss_list, test_acc_list, data_dir):
    log_path = f"{data_dir}/result.xlsx"
    all_data = []
    for iters, (loss, acc) in enumerate(zip(test_loss_list, test_acc_list), start=start_round + 1):
        data = [iters, round(loss.item(), 4), acc]
        all_data.append(data)
    df = pd.DataFrame(all_data, columns=['Iteration', 'Test Loss', 'Accuracy'])

    # 创建 ExcelWriter 对象
    with pd.ExcelWriter(log_path) as writer:
        # 将第一个 DataFrame 写入表格 'Sheet{i + 1}'
        df.to_excel(writer, index=False)
