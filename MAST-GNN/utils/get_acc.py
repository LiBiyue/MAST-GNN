import numpy as np
from sklearn.metrics import f1_score


def get_acc(pred, true, logger):
    acc_all, acc_all_H, acc_all_N, acc_all_L = [], [], [], []
    f1allscore = []
    for step in range(pred.shape[0]):
        acc_step, acc_step_H, acc_step_N, acc_step_L = [], [], [], []
        f1scores = []
        for idx in range(pred.shape[-1]):
            pred_node, true_node = pred[step, :, idx].reshape(-1), true[step, :, idx].reshape(-1)  # 变成向量 n*1
            pred_cls, true_cls = np.zeros(shape=pred_node.size), np.zeros(shape=pred_node.size)
            high_idx, normal_idx, low_idx = [], [], []
            for i in range(pred_node.size):
                if pred_node[i] < 2 / 3:
                    pred_cls[i] = 0
                    low_idx.append(i)
                elif pred_node[i] >= 4 / 3:
                    pred_cls[i] = 2
                    high_idx.append(i)
                else:
                    pred_cls[i] = 1
                    normal_idx.append(i)
            for i in range(true_node.size):
                if true_node[i] < 2 / 3:
                    true_cls[i] = 0
                elif true_node[i] >= 4 / 3:
                    true_cls[i] = 2
                else:
                    true_cls[i] = 1

            macro = f1_score(pred_cls, true_cls, average='macro')
            acc = sum(pred_cls == true_cls) / len(true_cls)
            accH = sum(true_cls[high_idx] == pred_cls[high_idx]) / len(true_cls[high_idx]) if high_idx else 0  # 高中 低
            accN = sum(true_cls[normal_idx] == pred_cls[normal_idx]) / len(true_cls[normal_idx]) if normal_idx else 0
            accL = sum(true_cls[low_idx] == pred_cls[low_idx]) / len(true_cls[low_idx]) if low_idx else 0

            acc_step.append(acc)
            acc_step_H.append(accH)
            acc_step_N.append(accN)
            acc_step_L.append(accL)
            f1scores.append(macro)
        acc_all.append(sum(acc_step) / len(acc_step))
        acc_step_H, acc_step_N, acc_step_L = \
            list(filter(lambda x: x != 0, acc_step_H)), list(filter(lambda x: x != 0, acc_step_N)), list(
                filter(lambda x: x != 0, acc_step_L))
        acc_step_H_, acc_step_N_, acc_step_L_ = \
            sum(acc_step_H) / len(acc_step_H) if acc_step_H else 0, \
            sum(acc_step_N) / len(acc_step_N) if acc_step_N else 0, \
            sum(acc_step_L) / len(acc_step_L) if acc_step_L else 0
        f1all = sum(f1scores) / len(f1scores)

        acc_all_H.append(acc_step_H_)
        acc_all_N.append(acc_step_N_)
        acc_all_L.append(acc_step_L_)
        f1allscore.append(f1all)

        logger.info(
            f'Horizon {step + 1:02d} {sum(acc_step) / len(acc_step):.4f}|{acc_step_H_:.4f}|{acc_step_N_:.4f}|{acc_step_L_:.4f} |f1score {f1all:.4f}')

    acc_all_H, acc_all_N, acc_all_L = list(filter(lambda x: x != 0, acc_all_H)), list(
        filter(lambda x: x != 0, acc_all_N)), list(filter(lambda x: x != 0, acc_all_L))
    acc_all_H_, acc_all_N_, acc_all_L_ = \
        sum(acc_all_H) / len(acc_all_H) if acc_all_H else 0, \
        sum(acc_all_N) / len(acc_all_N) if acc_all_N else 0, \
        sum(acc_all_L) / len(acc_all_L) if acc_all_L else 0
    flscore_all = sum(f1allscore) / len(f1allscore)
    logger.info(
        f'=Average=  {sum(acc_all) / len(acc_all):.4f} {acc_all_H_:.4f} {acc_all_N_:.4f} {acc_all_L_:.4f} |f1score {flscore_all:.4f}')