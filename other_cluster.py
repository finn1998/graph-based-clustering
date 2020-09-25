from evaluation import evaluate
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize as L2
import glob
import os

######################################################################################## combine 2 similarity matrix #####################################
# def sim_matrix(cnn, nodehis, param):
#     def normalize(feature):
#         sim = cosine_similarity(feature)
#         sim -= sim.min()
#         sim /= sim.max()
#         sim = 1-sim
#         return sim
#     cnn_sim = normalize(cnn)
#     nodehis_sim = normalize(nodehis)
#     sim = param*cnn_sim + (1-param)*nodehis_sim
#     return sim

# # input feature and gen sim: combine of both cnn sim matrix and node histogram sim matrix
# metrics = ['bcubed', 'nmi']
# cnn_path = glob.glob(r'/home/finn/research/layout_ocr/data/mizuho/all_images/*')
# cnn_path = glob.glob(r'/home/finn/research/data/clustering_data/mr_test/images/*')

# cnn_path = [os.path.split(path)[-1].split('.')[0] for path in cnn_path]
cnn_feature = np.fromfile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/mizuho_test_cnn.bin',dtype = np.float32)
cnn_feature = cnn_feature.astype(np.float32).reshape(-1,128)
cnn_feature /= np.sqrt((cnn_feature**2).sum(axis=1).reshape(cnn_feature.shape[0],1))
print(cnn_feature.shape)
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/mizuho_test_cnn.meta', 'r') as f:
    cnn_gr = f.readlines()
    cnn_gr = [int(l.replace('\n','')) for l in cnn_gr]

# nodehis_path = glob.glob(r'/home/finn/research/layout_ocr/data/mizuho/all_labels/*.json')
# nodehis_path = glob.glob(r'/home/finn/research/data/clustering_data/mr_test/labels/*.json')

# nodehis_path = [os.path.split(path)[-1].split('.')[0] for path in nodehis_path]

nodehis_feature = np.fromfile('/mnt/ai_filestore/home/finn/learn-to-cluster/data/features/mizuho_test_nodehis.bin',dtype = np.float32)
nodehis_feature = nodehis_feature.astype(np.float32).reshape(cnn_feature.shape[0],-1)
nodehis_feature /= np.sqrt((nodehis_feature**2).sum(axis=1).reshape(nodehis_feature.shape[0],1))
print(nodehis_feature.shape)
clustering = DBSCAN(eps=0.1, min_samples=1).fit(nodehis_feature).labels_
with open('/mnt/ai_filestore/home/finn/learn-to-cluster/data/labels/mizuho_test_nodehis.meta', 'r') as f:
    nodehis_gr = f.readlines()
    nodehis_gr = [int(l.replace('\n','')) for l in nodehis_gr]

# 
# index_dic = {}
# new_nodehis_gr = [0]*len(nodehis_gr)
# new_nodehis_feature = np.empty(nodehis_feature.shape, dtype= np.float32)
# for index,path in enumerate(nodehis_path):
#     index_dic[index] = cnn_path.index(path)
#     new_nodehis_gr[cnn_path.index(path)] = nodehis_gr[index]
#     new_nodehis_feature[cnn_path.index(path)] = nodehis_feature[index]
# import pdb
# pdb.set_trace()

# sim = sim_matrix(cnn_feature, new_nodehis_feature, 0.5)
dist = DistanceMetric.get_metric('euclidean')
cnn = dist.pairwise(cnn_feature)
node = dist.pairwise(nodehis_feature)
evaluate(np.asarray(cnn_gr), np.asarray(DBSCAN(eps=0.95, min_samples=1, metric = 'euclidean').fit(nodehis_feature).labels_), 'bcubed')
evaluate(np.asarray(cnn_gr), np.asarray(DBSCAN(eps=0.95, min_samples=1, metric = 'precomputed').fit(node).labels_), 'bcubed')
evaluate(np.asarray(cnn_gr), np.asarray(DBSCAN(eps=0.85, min_samples=1, metric = 'precomputed').fit(0.2*cnn+0.8*node).labels_), 'bcubed')
import pdb
pdb.set_trace()
th = [0.1, 0.2,0.3,0.4,0.6,0.8,0.9]
# th = [0.2,0.25, 0.3, 0.35,0.4]
for t in th:
    for i in metrics:
        evaluate(np.asarray(cnn_gr), np.asarray(DBSCAN(eps=t, min_samples=1, metric = 'precomputed').fit(sim).labels_), i)
# evaluate(np.asarray(gr), np.asarray(DBSCAN(eps=0.5, min_samples=1).fit(feature).labels_), 'nmi')
evaluate(np.asarray(cnn_gr), np.asarray(DBSCAN(eps=0.25, min_samples=1, metric = 'precomputed').fit(sim).labels_), 'bcubed')
# exit()
import pdb
pdb.set_trace()
pred = np.asarray(DBSCAN(eps=0.1, min_samples=1).fit(feature).labels_)
# pred_labels = DBSCAN(eps=0.8, min_samples=1).fit(feature).labels_
knn_labels = []
knn_feature = []
for index,label in enumerate(gr):
    if label not in knn_labels:
        knn_labels.append(label)
        knn_feature.append(feature[index])
knn_feature = np.asarray(knn_feature)
def knn(vec, features):
    return np.argsort(((features-vec)**2).sum(axis=1).reshape(features.shape[0]))[0]
count = 0
for index,feat in enumerate(feature):
    if knn_labels[knn(feat,knn_feature)] == gr[index]:
        count += 1
print(count)
print(feature.shape)


import shutil
import pdb
import glob
import os
train_path = r'/home/finn/research/layout_ocr/data/toyota4/all_images/*'
output = os.path.join(r'/home/finn/research/layout_ocr/data/toyota4','GCN_DBSCAN_toyota')
# train_path = r'/home/finn/research/data/clustering_data/mr_test/images/*'
# output = os.path.join(r'/home/finn/research/data/clustering_data/mr_test','NODEHIS_DBSCAN_mr')
if os.path.exists(output):
    shutil.rmtree(output)
os.mkdir(output)
# pdb.set_trace()

for label in set(pred):
    if not os.path.exists(os.path.join(output,f'cluter_{label}')):
        os.mkdir(os.path.join(output,f'cluter_{label}'))
for index,image in enumerate(glob.glob(train_path)):
    shutil.copy2(image, os.path.join(os.path.join(output,f'cluter_{pred[index]}'), os.path.split(image)[-1]))
# import pdb
# pdb.set_trace()
# import json
# import os
# import pdb 
# exit()
# pdb.set_trace()
# img_labels = json.load(open(r'/home/finn/research/data/clustering_data/test_index.json','r', encoding = 'utf-8'))
# import shutil
# output = r'/home/finn/research/data/clustering_data/mr_dbscan_output'
# for label in set(pred_labels):
#     if not os.path.exists(os.path.join(output,f'cluter_{label}')):
#         os.mkdir(os.path.join(output,f'cluter_{label}'))
# for image in img_labels:
#     shutil.copy2(image, os.path.join(os.path.join(output,f'cluter_{pred_labels[img_labels[image]]}'), os.path.split(image)[-1]))
# import pdb
# pdb.set_trace()