#! coding=utf8

import numpy as np
# import matplotlib.pyplot as plt

class Uitls(object):
  @staticmethod
  def sigmoid(x):
    return 1/(1 + np.exp(-x))

  # @staticmethod
  # def plotfit(weight, data, label):
  #   xcord1 = []
  #   ycord1 = []
  #   xcord2 = []
  #   ycord2 = []
  #   for i in range(n):
  #     if label[i] == 1:
  #       xcord1.append(data[i][1])
  #       ycord1.append(label[i][1])
  #     else:
  #       xcord2.append(data[i][1])
  #       ycord2.append(label[i][1])
  #   fig = plt.figure()
  #   ax = fig.add_subplot(111)
  #   ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
  #   ax.scatter(xcord2, ycord2, s=30, c='green')
  #   x = np.arange(-3, 3, 0.1)
  #   y = (- weight[0, 0] - weight[1, 0]*x)/weight[2, 0]
  #   ax.plot(x, y)
  #   plt.xlabel('x1')
  #   plt.ylabel('x2')
  #   plt.show()

  @staticmethod
  def grad_ascent(data, label):
    data = np.mat(data)
    label = np.mat(label).transpose()
    m, n = np.shape(data)
    weight = np.ones((n, 1))
    alpha = 1e-3
    maxcycle = 300

    for i in range(maxcycle):
      h = Uitls.sigmoid(data * weight)
      error = label - h
      weight += alpha * data.T * error
    return weight

  @staticmethod
  def stoc_grad_ascent(data, label):
    m, n = np.shape(data)
    alpha = 1e-3
    weight = np.ones(n)
    for i in range(m):
      h = Uitls.sigmoid(sum(data[i] * weight))
      error = label[i] - h
      weight += alpha * error * data[i]
    return weight

  @staticmethod
  def stoc_grad_ascent_batch(data, label, batch):
    m, n = np.shape(data)
    weight = np.ones(n)
    interval_value = int(m/batch)+1
    for i in range(batch):
      temp_data = data[i::interval_value]
      temp_label = label[i::interval_value]
      for j in range(m):
        alpha = 4 / (1 + i + j) + 1e-3
        randindex = int(np.random.uniform(0, m))
        h = Uitls.sigmoid(sum(temp_data[i] * weight))
        error = temp_label[i] - h
        weight += alpha * error * temp_data[i]
    return weight

  @staticmethod
  def batch_data(data, label, batch_size):
    m, n = np.shape(data)
    group_num = int(m/batch_size)
    for i in range(group_num):
      temp_data = data[i::batch_size]
      temp_label = label[i::batch_size]
      yield temp_data, temp_label

  @staticmethod
  def split_train_test_data(data, label, train_rate):
    m, n = np.shape(data)
    np.random.seed(7)
    mask_tmp = np.random.choice([0, 1], m, p=[1-train_rate, train_rate])
    train_data = data[mask_tmp>0]
    train_label = label[mask_tmp>0]
    test_data = data[mask_tmp==0]
    test_label = label[mask_tmp==0]
    return train_data, train_label, test_data, test_label

  @staticmethod
  def split_train_test_data_ROW(data, label, train_rate):
    m, n = np.shape(data)
    train_data_row = m * train_rate
    l = range(m)
    train_mask = np.random.choice(l, train_data_row, replace=False)
    test_mask = np.array(list(set(l) - set(train_mask)))
    np.random.seed(7)
    train_data = data[train_mask]
    train_label = label[train_mask]
    test_data = data[test_mask]
    test_label = label[test_mask]
    return train_data, train_label, test_data, test_label 

def test():
  data_len = 50
  data_wide = 10
  batch_size = 2
  test_data = np.random.randint(1, 10, (data_len, data_wide))
  label_data = np.random.randint(0, 2, (data_len, 1))
  # for data, label in Uitls.batch_data(test_data, label_data, batch_size):
  #   print(data, label)
  train_D, train_T, test_D, test_T = Uitls.split_train_test_data(test_data, label_data, 0.7)
  model_weight = Uitls.stoc_grad_ascent(train_D, train_T)
  predict_T = Uitls.sigmoid(sum(test_D * model_weight))
  print([i for i in zip(test_T, predict_T)])


if __name__ == '__main__':
  test()









