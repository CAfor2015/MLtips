#! coding=utf8
import numpy as np
from functools import reduce

# CART(classification and regerssion trees) detail

class Node(object):
  def __init__(self, split_value=None, L_branch=None, R_branch=None, 
        results=None, col=-1, summary=None, data=None):
    self.left = L_branch
    self.right = R_branch
    self.results = results
    self.col = col
    self.split_value = split_value
    self.summary = summary
    self.data = data

  def _node_judge(self, data):
    if self.results is not None:
      key_len = len(self.results.keys())
      if key_len == 1:
        return self.results.keys()
      elif key_len > 1:
        len_item = reduce(lambda x, y: x+y, self.results.values())
        return dict(zip(self.results.keys(), map(lambda x: x/len_item, self.results.values())))
    else:
      judge_value = data[self.col]
      if judge_value >= self.split_value:
        return self.left._node_judge(data)
      else:
        return self.right._node_judge(data)

  def _mid_show(self, depth=1):
    if self.left is not None:
      self.left._mid_show(depth+1)
    if self.col >= -1:
      print("**--"*depth, self)
    if self.right is not None:
      self.right._mid_show(depth+1)

  def __repr__(self):
    return "col:%s, split_value:%s, results:%s, summary:%s, data:%s\n"% \
          (self.col, self.split_value, self.results, self.summary, self.data)

class CARTree(object):
  """docstring for CARTree"""
  def __init__(self, values=None):
    self.values = values
    self.root = self._buildDecisionTree(values=self.values)

  def _show(self):
    self.root._mid_show()

  def _layer_show(self):
    temp_list = []
    def _layer_info(node, depth=1):
      while len(temp_list) > 0:
        print(temp_list.pop())
      temp_list.append("**--"*depth+str(node))
      if node.left is not None:
        _layer_info(node.left, depth+1)
      if node.right is not None:
        _layer_info(node.right, depth+1)
    if self.root is not None:
      _layer_info(self.root)


  def entropy(self, label_set):
    return -np.sum(label_set * np.log2(label_set))

  def _calculateDiffCount(self, datas):
    values = datas.ravel()
    items = set(values)
    info_dict = dict(zip(items, [len(np.argwhere(values==i).ravel()) for i in items]))
    return info_dict

  def _gini(self, rows):
    row_value = rows[:,-1].ravel()
    if len(row_value) < 1:
      return 0.0
    lenght_sqr = len(row_value)**2
    results = self._calculateDiffCount(row_value)
    imp = reduce(lambda a, b: a+b, map(lambda x: x**2/lenght_sqr, results.values()))
    return 1 - imp

  def _buildDecisionTree(self, values=None):
    # build decision tree by recursive function
    # stop recursive function when gain = 0
    if len(values) == 0:
      return None
    # if len(values) == 1:
    #   return Node(results=self._calculateDiffCount(values[:,-1]), summary=None, data=values)
    values_gini = self._gini(values)
    shape_info = values.shape
    column_lenght = shape_info[1]
    rows_lenght = shape_info[0]

    best_diffgini = 0.0
    best_value = None
    best_set = None

    # choose best diffgain
    for col in range(column_lenght - 1):
      col_value = values[:, col]
      col_value_set = set(col_value)
      for split_point in col_value_set:
        list1, list2 = values[col_value >= split_point], values[col_value < split_point]
        p = len(list1)/values.shape[0]
        diffgini = values_gini - (p * self._gini(list1) + (1-p) * self._gini(list2))
        if diffgini > best_diffgini:
          best_diffgini = diffgini
          best_value = (col, split_point)
          best_set = (list1, list2)
    discrp = {'impurity': '%.3f'%values_gini, 'sample': '%d'%rows_lenght}

    if best_diffgini > 0.0:
      right_branch_node = self._buildDecisionTree(values=best_set[0])
      left_branch_node = self._buildDecisionTree(values=best_set[1])
      return Node(col=best_value[0], split_value=best_value[1],
          L_branch=left_branch_node, R_branch=right_branch_node, summary=discrp)
    elif best_diffgini == 0.0:
      return Node(results=self._calculateDiffCount(values[:,-1]), summary=discrp, data=values)

  def prune(self, mindiffGini):
    def _node_merge(node, mindiffGini):
      if node.left.results is None:
        _node_merge(node.left, mindiffGini)
      if node.right.results is None:
        _node_merge(node.right, mindiffGini)
      if node.left.results is not None and node.right.results is not None:
        print("cutting branch")
        len1 = len(node.left.data)
        len2 = len(node.right.data)
        merge_data = np.vstack((node.left.data, node.right.data))
        p = float(len1)/(len1 + len2)
        diffgain = self._gini(merge_data)-(p*self._gini(node.right.data)+(1-p)*self._gini(node.left.data))
        if diffgain < mindiffGini:
          print("cutting branch opration")
          node.data = merge_data
          node.results = self._calculateDiffCount(node.data[:,-1])
          node.col = -1
          node.left = None
          node.right = None
    if self.root is not None:
      _node_merge(self.root, mindiffGini)

  def classify(self, data):
    if self.root is not None:
      return self.root._node_judge(data)
    else:
      raise("tree has not build yet")

def test():
  np.random.seed(9)
  test_values_rows = 16
  info_values = np.random.randint(10, 20, (test_values_rows, 10))
  class_values = np.random.randint(1, 6, (test_values_rows, 1))
  test_values = np.hstack((info_values, class_values))
  print(test_values)
  cart = CARTree(test_values)
  cart._show()
  cart.prune(0.44)
  print("#"*30," after prune off ","#"*30,"\n")
  cart._show()
  test_classify_value = np.random.randint(10, 20, (1, 10))
  print("predict class:", cart.classify(test_classify_value.ravel()))


if __name__ == '__main__':
  test()
