#! coding=utf-8

import numpy as np
from functools import reduce
np.random.seed(7)
import time

class Activation(object):
  @staticmethod
  def sigmoid(x):
    return 1.0/(1+np.exp(-x))

  @staticmethod
  def sigmoid_differential(x):
    temp = 1.0/(1+np.exp(-x))
    return temp(1 - temp)

  @staticmethod
  def tanh(x):
    return np.tanh(x)

  @staticmethod
  def tanh_differential(x):
    return 1.0-np.tanh(x)**2


class Utils(object):
  @staticmethod
  def softmax(value):
    temp = reduce(lambda x, y: x+y, value)
    return np.array([i for i in map(lambda x: x/temp, value)])

  @staticmethod
  def normaliz(value):
    std_value = np.std(value)
    mean_value = np.mean(value)
    nor = map(lambda x: (x-mean_value)/std_value, value)[0]
    max_value = max(abs(nor))
    if max_value > 1:
      return nor/(max_value+1e-6)
    else:
      return nor

  @staticmethod
  def normaliz_n_dim(value):
    row_sum = value.sum(axis=1)
    new_matrix = value/row_sum[:, np.newaxis]
    return new_matrix - np.mean(new_matrix, axis=1)[:, np.newaxis]


class Lossfunction(object):
  @staticmethod
  def cross_entropy(target, out_value):
    s_target, s_out = [i for i in map(lambda x: Utils.softmax(x), [target, out_value])]
    # cross_entropy_value = sum(map(lambda x: -x[0]*np.log2(x[1]), zip(s_target, s_out)))
    cross_entropy_value = 0
    for a, b in zip(s_target, s_out):
      cross_entropy_value += - a*np.log2(b)

    # clipping error value
    if cross_entropy_value > 5:
      return 2.0
    return cross_entropy_value


class NeuralNetwork(object):
  def __init__(self, layers, activation='tanh', init_value=0.25):
    # leyers is a list that contain the number of neural in each layer
    self.activation = getattr(Activation, activation)
    self.activation_diff = getattr(Activation, activation+'_differential')
    self.weight = []
    for i in range(1, len(layers)):
      self.weight.append((2*np.random.random((layers[i-1], layers[i]))-1)*init_value)

  def fit(self, x, y, learning_rate=2e-3, batch_size=1, epoch=1):
    x = np.atleast_2d(x)
    x = Utils.normaliz_n_dim(x)
    row_num = x.shape[0]
    batch_time = int(row_num/batch_size)
    temp_error = 100
    for k in range(batch_time):
      i = np.random.randint(row_num)
      a = [x[i].T]

      # forward
      for l in range(len(self.weight)):
        # print(a[l].shape, self.weight[l].shape)
        a.append(self.activation(np.dot(a[l], self.weight[l])))
      out_error = Lossfunction.cross_entropy(y[i].ravel(), a[-1].ravel())
      deltas = [out_error*self.activation_diff(a[-1])]
      # print("out_error:", self.activation_diff(out_error).shape, a[-1].shape)
      deltas = [np.dot(self.activation_diff(out_error).T, a[-1].T)]

      # backward
      for i in range(len(a)-2, 0, -1):
        # print(type(deltas[0]), self.activation_diff(deltas[0].T), deltas[0].shape, self.weight[i].shape)
        if i == len(a) - 2:
          deltas.insert(0, np.dot(deltas[0].T, self.weight[i].T)*self.activation_diff(a[i]))
        else:
          deltas.insert(0, np.dot(deltas[0], self.weight[i].T)*self.activation_diff(a[i]))
      
      # updata weight value
      for i in range(len(self.weight)):
        layer = np.atleast_2d(a[i])
        delta = np.atleast_2d(deltas[i])
        self.weight[i] += learning_rate * np.dot(layer.T, delta)

      # stop or change learning_rate
      if temp_error < 1.5849625007211564:
        learning_rate = learning_rate*1e-1
      # if out_error < 1.33:
      #   return

      # info and sleep
      if k & 0x0F == 0x0F:
        temp_error = out_error
        print("elements:", k, "err:", out_error, "weight:", a[0])
      time.sleep(5e-3)

  def predict(self, value):
    x = np.array(value.ravel())
    temp = np.dot(x.T, self.weight[0])
    for l in range(1, len(self.weight)):
      temp = self.activation(np.dot(temp, self.weight[l]))
    return temp


def test_tool_function():
  test_value = np.random.randint(1, 10, (10, 5))
  print(test_value)
  print(Utils.normaliz_n_dim(test_value))
  test_entropy = np.zeros((10))
  test_entropy[4] = 1
  test_entropy_value = np.random.randint(1, 10, 10)
  print(test_entropy, test_entropy_value)
  print(Lossfunction.cross_entropy(test_entropy, test_entropy_value))

def test():
  input_shape = 5
  output_shape = 3
  value_rows = 100000
  train_value = np.random.randint(1, 10, (value_rows, input_shape))
  tag_info = np.random.randint(0, output_shape, (value_rows, 1))
  train_tag = np.zeros((value_rows, output_shape))
  for i in range(value_rows):
    train_tag[i][tag_info[i]] = 1
  print(len(np.argwhere(train_tag==1)))
  nn = NeuralNetwork(layers=[input_shape, 8, 5, output_shape], activation='sigmoid', init_value=0.5)
  nn.fit(x=train_value, y=train_tag, learning_rate=1e-4, batch_size=1)
  [print(i,i.shape) for i in nn.weight]
  test_value = 2*np.random.random((1, input_shape))-1
  print(nn.predict(test_value))


if __name__ == '__main__':
  # test_tool_function()
  test()


