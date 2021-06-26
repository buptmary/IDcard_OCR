#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
from sklearn.metrics import accuracy_score
from torch.autograd import Variable

from pic2mnist import CutIdCard
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# 超参数
EPOCH = 4
BATCH_SIZE = 32
LR = 0.001
DOWNLOAD_MNIST = False  # 若数据集已经存在则不用下载
IFTRAIN = False  # 若模型已经保存则不用训练
torch.manual_seed(1)


# 定义网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(32, 128, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(128 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


# 获得训练好的CNN模型
def get_model(iftrain):
    if iftrain:
        train_data = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=DOWNLOAD_MNIST,
        )

        # 构建dataloader
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

        cnn = CNN()
        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
        loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

        # 训练
        losses = []
        for epoch in range(EPOCH):
            for step, (b_x, b_y) in enumerate(train_loader):
                output = cnn(b_x)
                loss = loss_func(output, b_y)
                if step % 50 == 0:
                    losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        plt.plot(losses)
        plt.show()
        torch.save(cnn, '/model/mnist_cnn.pt')
    else:
        cnn = torch.load('D:/ocr/model/mnist_cnn.pt')
    return cnn


def main():
    path = r"IDpic/ID1.png"
    CutIdCard(path)
    print('图片处理完成！')
    cnn = get_model(IFTRAIN)
    print('加载CNN模型成功！')

    # 读取图片
    x = []
    for i in range(18):
        img = cv.imread(r"D:/ocr/tmp/pic/"+str(i)+".png")
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, 0)
        x.append(img)

    # 进行预测
    x = Variable(torch.from_numpy(np.array(x)))
    output = cnn(x)
    pred =torch.max(output, 1)[1].data.numpy()
    pred = [str(i) for i in pred]
    print("加载预测结果为：")
    print('身份证号为：'+''.join(pred))


if __name__ == '__main__':
    main()
