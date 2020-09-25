import os.path as osp
import logging

# data locations
prefix = './data'
# train_name = 'part0_train'
train_name = 'mr_train_nodehis'
# test_name = 'part1_test'
test_name = 'mr_test_nodehis'
knn = 80
knn_method = 'faiss'
th_sim = 0.  # cut edges with similarity smaller than th_sim

# if `knn_graph_path` is not passed, it will build knn_graph automatically
train_data = dict(feat_path=osp.join(prefix, 'features',
                                    '{}.bin'.format(train_name)),
                 label_path=osp.join(prefix, 'labels',
                                     '{}.meta'.format(train_name)),
                 knn_graph_path=osp.join(prefix, 'knns', train_name,
                                         '{}_k_{}.npz'.format(knn_method,
                                                              knn)),
                 k=knn,
                 is_norm_feat=True,
                 th_sim=th_sim,
                 conf_metric='s_nbr')

test_data = dict(feat_path=osp.join(prefix, 'features',
                                    '{}.bin'.format(test_name)),
                 label_path=osp.join(prefix, 'labels',
                                     '{}.meta'.format(test_name)),
                 knn_graph_path=osp.join(prefix, 'knns', test_name,
                                         '{}_k_{}.npz'.format(knn_method,
                                                              knn)),
                 k=knn,
                 is_norm_feat=True,
                 th_sim=th_sim,
                 conf_metric='s_nbr')

# model
model = dict(type='gcn_v',
             kwargs=dict(feature_dim=128, nhid=512, nclass=1, dropout=0.))

# training args
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-5)
optimizer_config = {}

# total_epochs = 80000
total_epochs = 10000
lr_config = dict(
    policy='step',
    step = [int(r * total_epochs) for r in [0.5, 0.8, 0.9]]
)

batch_size_per_gpu = 1
workflow = [('train_gcnv', 1)]

# testing args
use_gcn_feat = True
max_conn = 1
tau_0 = 0.75
tau = 0.8

metrics = ['pairwise', 'bcubed', 'nmi']

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=500)

log_level = logging.getLogger("some.logger")#'INFO'
log_config = dict(interval=1, hooks=[
    dict(type='TextLoggerHook'),
])
