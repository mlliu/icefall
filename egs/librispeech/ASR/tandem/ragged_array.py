import k2.ragged as k2r
import torch

# shape = k2r.RaggedShape('[[x x] [x]]')
# print('shape:', shape)
# print('num_axes:', shape.num_axes)
# print('row_splits:', shape.row_splits(1))
# print('row_ids:', shape.row_ids(1))
# print('row_splits(2):', shape.row_splits(2))
# print('row_ids(2):', shape.row_ids(2))

# k2r.create_ragged_shape2(shape.row_splits(1), shape.row_ids(1))

#create a ragged tensor
# values = torch.tensor([1, 2, 3])
# shape = k2r.RaggedShape('[[x x] [x]]')
# ragged = k2r.RaggedTensor(shape, values)
# print('ragged:', ragged)
a = k2r.create_ragged_tensor([ [1], [], [2,3]])
print('a:', a)
print('a.values:', a.values)
print('a.num_axes:', a.num_axes)
print('a.shape:', a.shape)
print('a.shape.num_axes:', a.shape.num_axes)
print('a.shape.row_splits(1):', a.shape.row_splits(1))
print('a.shape.row_ids(1):', a.shape.row_ids(1))