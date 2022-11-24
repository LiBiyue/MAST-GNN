import argparse

import numpy as np
import torch


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    # cuda = True if torch.cuda.is_available() else False
    TensorFloat =  torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


def normalize_dataset(data):
    main_scaler = StandardScaler(data[..., -1].mean(), (data[..., -1]).std())
    data[..., -1] = main_scaler.transform(data[..., -1])

    for idx in range(data.shape[-1] - 1):
        feature_scaler = StandardScaler(data[..., idx].mean(), data[..., idx].std())
        data[..., idx] = feature_scaler.transform(data[..., idx])
    return data, main_scaler


def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets, scaler=None):
    """
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes, input_dim = data.shape

    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = np.expand_dims(data[t + y_offsets, :, -1], axis=-1)
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    raw_data = np.load(args.feature_data_path,mmap_mode ='r')
    raw_data=raw_data.astype(np.float)
    print(raw_data.shape,raw_data.dtype)
    # print(raw_data.shpae)
    data, scaler = normalize_dataset(raw_data)

    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(args.y_start, (seq_length_y + 1), 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(data, x_offsets, y_offsets)
    # per = np.random.permutation(x.shape[0])  # 打乱后的行号
    # x = x[per, :, :,:]  # 获取打乱后的训练数据
    # y = y[per, :, :,:]
    print('\n****************** Data Generator ******************')
    print(f'x shape: {x.shape}, y shape: {y.shape}')

    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_test - num_train
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = x[num_train: num_train+num_val], y[num_train: num_train+num_val]
    x_test, y_test = x[-num_test:], y[-num_test:]
    # print(x_train.shape,x[0,:,0,0])
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(f'{cat} x: {_x.shape}, y: {_y.shape}')

    train_dataloader = data_loader(x_train, y_train, args.batch_size, shuffle=True, drop_last=True)
    val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
    test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=True)
    return train_dataloader, val_dataloader, test_dataloader, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_data_path", type=str, default=r"D:\QQ\1392776996\FileRecv\features.npy")
    parser.add_argument("--seq_length_x", type=int, default=12)
    parser.add_argument("--seq_length_y", type=int, default=12)
    parser.add_argument("--y_start", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    train, val, test, scaler = generate_train_val_test(args)
    print(len(train.dataset))