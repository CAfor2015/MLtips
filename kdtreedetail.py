from operator import itemgetter
import numpy as np

class Node(object):
  def __init__(self, value, types=1, father=None):
    self.types = types    #1:branch, 2:leaf
    self.value = value
    self.father = father
    self.left = None
    self.right = None

  def get_types(self):
    return self.types

  def _mid_show(self):
    if self.left is not None:
      self.left._mid_show()
    if self.types == 2:
      print(self)
    if self.right is not None:
      self.right._mid_show()

  def _find(self, search_value, depth, dim_order):
    if self.types is 2:
      return self
    if search_value[dim_order[depth]] < self.value[dim_order[depth]]:
      if self.left is not None:
        return self.left._find(search_value, depth+1, dim_order)
      else:
        return self
    elif search_value[dim_order[depth]] >= self.value[dim_order[depth]]:
      if self.right is not None:
        return self.right._find(search_value, depth+1, dim_order)
      else:
        return self
  
  def __repr__(self):
    return "--->%s\n"%(self.value)


class Tree(object):
  def __init__(self, points):
    self.root = None
    if len(points) == 0:
      return None
    self.dim_order = np.argsort(np.std(points, axis=0))
    self.means = np.mean(points, axis=0)
    self.std = np.std(points, axis=0)
    self._kd_tree(points, 0)

  def _kd_tree(self, points, depth, father=None):
    # print(depth, points)
    if len(points) == 0:
      return None
    if len(points) == 1:
      return Node(points[0], types=2, father=father)
    if depth == len(self.dim_order):
      return None
    # choose max std column as cut dim
    cutting_dim = self.dim_order[depth]
    medium_index = len(points)//2
    stor_index = np.argsort(points[:,cutting_dim], axis=0)
    if depth < len(self.dim_order)-1:
      node = Node(points[medium_index,:], father=father)
    if depth == 0:
      self.root = node
    node.left = self._kd_tree(points[stor_index[:medium_index]], depth+1, father=node)
    node.right = self._kd_tree(points[stor_index[medium_index:]], depth+1, father=node)
    return node

  def _forward_search(self, value):
    if type(value) is Node:
      value = value.value
    if len(value) != len(self.dim_order):
      return (False, [], "error vector dimension")
    return self.root._find(value, 0, self.dim_order)

  def _backward_search(self, node):
    path_node = []
    path_choose = []
    def _father_value(node):
      path_node.append(node)
      if node.father is not None:
        _father_value(node.father)
      else:
        return
    _father_value(node)
    for i in range(len(path_node)-1):
      path_choose.append(0 if (path_node[i] is path_node[i+1].left) else 1)
    return [i for i in zip(path_node[::-1][:-1], path_choose[::-1])]

  def _findmin(n, depth, cutting_dim, min):
    if min is None:
      min = n.location
    if n is None:
      return min
    current_cutting_dim = depth%len(min)
    if n.location[cutting_dim] < min[cutting_dim]:
      min = m.location
    if cutting_dim == current_cutting_dim:
      return findmin(n, depth, cutting_dim, min)(n.left, depth+1, cutting_dim, min)
    else:
      leftmin = _findmin(n.left, depth+1, cutting_dim, min)
      rightmin = _findmin(n.right, depth+1, cutting_dim, min)
      if leftmin[cutting_dim] > rightmin[cutting_dim]:
        return rightmin
      else:
        return leftmin

  def _search_near(self, value):
    nodes = []
    leaf_node = self._forward_search(value)
    path_choose = self._backward_search(leaf_node)
    near_search_order = np.argsort(map(lambda x: float(x[0])/float(x[1]), zip(np.abs(value - self.means), self.std)))
    change_index = self.dim_order.tolist().index(near_search_order[0])
    nodes.append(path_choose[-1][0].left)
    nodes.append(path_choose[-1][0].right)
    if path_choose[-1][1] != path_choose[-2][1]:
      local_near = path_choose[-2][0]
      if path_choose[-2][1] is 0:
        local_near = local_near.right
        nodes.append(local_near.left)
      elif path_choose[-2][1] is 1:
        local_near = local_near.left
        nodes.append(local_near.right)
    tmp_node = path_choose[0][0]
    for node_path in path_choose[change_index:]:
      if node_path[1] is 0:
        tmp_node = tmp_node.right
      elif node_path[1] is 1:
        tmp_node = tmp_node.left
    nodes.append(tmp_node)
    return [i.value for i in nodes]

  def insert(self, point, depth):
    if self.root is None:
      return Node(point)
    cutting_dim = depth%(point)
    if point[cutting_dim] < n.location[cutting_dim]:
      if n.left is None:
        n.left = Node(point)
      else:
        insert(n.left, point, depth+1)
    else:
      if n.right is None:
        n.right = Node(point)
      else:
        insert(n.right, point, depth+1)

  def delete(n, point, depth):
    cutting_dim = depth % len(point)
    if n.location == point:
      if n.right is not None:
        n.location = findmin(n.right, depth+1, cutting_dim, None)
        delete(n.right, n.location, depth+1)
      elif n.left is not None:
        n.location == findmin(n.left, depth+1)
        delete(n.left, n.location, depth+1)
        n.right = n.left
        n.left = None
    else:
      if point[cutting_dim] < n.location[cutting_dim]:
        delete(n.left, point, depth+1)
      else:
        delete(n.right, point, depth+1)

def test():
  np.random.seed(7)
  test_tree_value = np.random.randint(1, 100, (16, 20))
  test_value_0 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
  dis_0 = [i for i in map(lambda x: np.linalg.norm(x - test_value_0), test_tree_value)]
  test_value_1 = np.array([5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])
  dis_1 = [i for i in map(lambda x: np.linalg.norm(x - test_value_1), test_tree_value)]
  test_value_2 = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10])
  dis_2 = [i for i in map(lambda x: np.linalg.norm(x - test_value_2), test_tree_value)]
  kd_tree_test = Tree(test_tree_value)
  # kd_tree_test.root._mid_show()
  l0 = kd_tree_test._search_near(test_value_0)
  l1 = kd_tree_test._search_near(test_value_1)
  l2 = kd_tree_test._search_near(test_value_2)
  for a, b in zip([test_value_0, test_value_1, test_value_2], [l0, l1, l2]):
    print(b)
    print([i for i in map(lambda x: np.linalg.norm(x - a), b)])
  print(dis_0)
  print(min(dis_0))
  print(dis_1)
  print(min(dis_1))
  print(dis_2)
  print(min(dis_2))

if __name__ == '__main__':
  test()

