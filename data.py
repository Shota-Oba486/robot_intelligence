import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from dataset.mnist import load_mnist
import numpy as np


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)
    return x_train, t_train,x_test,t_test

def applyRandomErrors(data, error_ratio):
    # 元のデータのコピーを作成して、エラーを含む行列を作成
    data_with_errors = data.astype(float).copy()
    # 0から1のランダムなエラー値を持つ行列を生成
    random_error_values = np.random.rand(data.shape[0], data.shape[1])
    # エラー割合に基づいてTrue/Falseを含むマスクを生成
    error_mask = np.random.choice([False, True], size=data.shape, p=[error_ratio, 1 - error_ratio])
    # マスクがFalseの位置にエラー値を代入
    data_with_errors[~error_mask] = random_error_values[~error_mask]
    
    return data_with_errors