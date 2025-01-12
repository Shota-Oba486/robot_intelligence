import numpy as np
import tqdm
from data import applyRandomErrors
from data import get_data

def train(model, optimizer,error_ratio=0):
    n_epoch = 15 # エポック数
    batchsize = 100
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    x_train,t_train,x_test,t_test = get_data()
    train_n, test_n = x_train.shape[0], x_test.shape[0]

    # for epoch in tqdm.tqdm(range(n_epoch)):
    for epoch in range(n_epoch):
        # train start
        pred_y = []
        sum_loss = 0
        x_train_error = applyRandomErrors(x_train,error_ratio)
        x_test_error = applyRandomErrors(x_test,error_ratio)

        for i in range (0,train_n,batchsize):
            x = x_train_error[i: i+batchsize]
            t = t_train[i: i+batchsize]
            
            loss = model.forward(x,t)
            model.backward()
            optimizer.update()
            sum_loss += loss * batchsize

            pred_y_in_each_batch = np.argmax(model.y,axis=1)
            pred_y = np.concatenate((pred_y, pred_y_in_each_batch)).astype(int) 
            
        # acuuracy：予測結果を１-hot表現に変換し、正解との要素積を取ることで、正解数を図る
        pred_y_1hot = np.eye(10)[pred_y]
        accuracy_each_epoch = np.sum(pred_y_1hot*t_train) / train_n
        train_acc_list.append(accuracy_each_epoch)
        loss_each_epoch = sum_loss / train_n
        train_loss_list.append(loss_each_epoch)

        # print(f"train:epoch:{epoch+1},acuuracy:{accuracy_each_epoch:.5f},loss:{sum_loss/train_n:.5f}   |  ",end = "")

        # test start

        pred_y_test = []
        sum_loss_test = 0
        for i in range (0,test_n,batchsize):
            x = x_test_error[i:i+batchsize]
            t = t_test[i:i+batchsize]
            
            loss = model.forward(x,t)
            pred_y_test_in_batch = np.argmax(model.y,axis=1)
            pred_y_test = np.concatenate((pred_y_test, pred_y_test_in_batch)).astype(int)

            sum_loss_test += model.loss * batchsize

        pred_y_test_1hot = np.eye(10)[pred_y_test]
        accuracy_test = np.sum(pred_y_test_1hot * t_test) / test_n
        test_acc_list.append(accuracy_test)
        test_loss_list.append(sum_loss_test/test_n)

        # print(f"test:accuracy:{accuracy_test:.5f},loss:{sum_loss_test/test_n:.5f}")
    
    return train_loss_list,train_acc_list,test_loss_list,test_acc_list