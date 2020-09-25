from evaluation import evaluate
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize as L2
import glob
import os

######################################################################################## combine 2 similarity matrix #####################################

cnn_path = glob.glob(r'/home/finn/research/layout_ocr/data/toyota4/all_images/*')
# cnn_path = glob.glob(r'/home/finn/research/data/clustering_data/mr_test/images/*')

cnn_path = [os.path.split(path)[-1].split('.')[0] for path in cnn_path]
cnn_feature = np.fromfile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/toyota_raw.bin')
cnn_feature = cnn_feature.astype(np.float32).reshape(-1,128)
cnn_feature /= np.sqrt((cnn_feature**2).sum(axis=1).reshape(cnn_feature.shape[0],1))
print(cnn_feature.shape)
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/toyota_raw.meta', 'r') as f:
    cnn_gr = f.readlines()
    cnn_gr = [int(l.replace('\n','')) for l in cnn_gr]

nodehis_path = glob.glob(r'/home/finn/research/layout_ocr/data/toyota4/all_labels/*.json')
# nodehis_path = glob.glob(r'/home/finn/research/data/clustering_data/mr_test/labels/*.json')

nodehis_path = [os.path.split(path)[-1].split('.')[0] for path in nodehis_path]

nodehis_feature = np.fromfile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/toyota_nodehis.bin',dtype = np.float32)
nodehis_feature = nodehis_feature.astype(np.float32).reshape(cnn_feature.shape[0],-1)
nodehis_feature /= np.sqrt((nodehis_feature**2).sum(axis=1).reshape(nodehis_feature.shape[0],1))
print(nodehis_feature.shape)
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/toyota_nodehis.meta', 'r') as f:
    nodehis_gr = f.readlines()
    nodehis_gr = [int(l.replace('\n','')) for l in nodehis_gr]

# 
index_dic = {}
new_nodehis_gr = [0]*len(nodehis_gr)
new_nodehis_feature = np.empty(nodehis_feature.shape, dtype= np.float32)
for index,path in enumerate(nodehis_path):
    index_dic[index] = cnn_path.index(path)
    new_nodehis_gr[cnn_path.index(path)] = nodehis_gr[index]
    new_nodehis_feature[cnn_path.index(path)] = nodehis_feature[index]

test_path = sorted(glob.glob(r'/home/finn/research/layout_ocr/data/toyota4/test/images/**/*'))
test_path = [os.path.split(path)[-1].split('.')[0] for path in test_path]
train_path = sorted(glob.glob(r'/home/finn/research/layout_ocr/data/toyota4/train/images/**/*'))
train_path = [os.path.split(path)[-1].split('.')[0] for path in train_path]
########################################### nodehis #############
test_nodehis_gr = [0]*len(test_path)
train_nodehis_gr = [0]*len(train_path)
test_nodehis_feature = np.empty((len(test_path), nodehis_feature.shape[1]), dtype= np.float32)
train_nodehis_feature = np.empty((len(train_path), nodehis_feature.shape[1]), dtype= np.float32)

for idx,path in enumerate(cnn_path):
    if path in test_path:
        test_nodehis_gr[test_path.index(path)] = cnn_gr[idx]
        test_nodehis_feature[test_path.index(path)] = new_nodehis_feature[idx]
    else:
        train_nodehis_gr[train_path.index(path)] = cnn_gr[idx]
        train_nodehis_feature[train_path.index(path)] = new_nodehis_feature[idx]
test_nodehis_feature = np.asarray(test_nodehis_feature).astype(np.float32)
train_nodehis_feature = np.asarray(train_nodehis_feature).astype(np.float32)
test_nodehis_feature.tofile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/toyota_test_nodehis.bin')
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/toyota_test_nodehis.meta','w') as f:
    for line in test_nodehis_gr:
        f.write('{}\n'.format(line))

train_nodehis_feature.tofile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/toyota_train_nodehis.bin')
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/toyota_train_nodehis.meta','w') as f:
    for line in train_nodehis_gr:
        f.write('{}\n'.format(line))

############################################# cnn ###########################
test_nodehis_gr = [0]*len(test_path)
train_nodehis_gr = [0]*len(train_path)
test_nodehis_feature = np.empty((len(test_path), cnn_feature.shape[1]), dtype= np.float32)
train_nodehis_feature = np.empty((len(train_path), cnn_feature.shape[1]), dtype= np.float32)

for idx,path in enumerate(cnn_path):
    if path in test_path:
        test_nodehis_gr[test_path.index(path)] = cnn_gr[idx]
        test_nodehis_feature[test_path.index(path)] = cnn_feature[idx]
    else:
        train_nodehis_gr[train_path.index(path)] = cnn_gr[idx]
        train_nodehis_feature[train_path.index(path)] = cnn_feature[idx]
test_nodehis_feature = np.asarray(test_nodehis_feature).astype(np.float32)
train_nodehis_feature = np.asarray(train_nodehis_feature).astype(np.float32)
test_nodehis_feature.tofile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/toyota_test_cnn.bin')
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/toyota_test_cnn.meta','w') as f:
    for line in test_nodehis_gr:
        f.write('{}\n'.format(line))

train_nodehis_feature.tofile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/toyota_train_cnn.bin')
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/toyota_train_cnn.meta','w') as f:
    for line in train_nodehis_gr:
        f.write('{}\n'.format(line))

