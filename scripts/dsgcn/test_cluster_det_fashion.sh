config=./dsgcn/configs/cfg_test_det_fashion_20_prpsls.py
load_from=./data/pretrained_models/pretrained_gcn_d_fashion.pth

export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=. python dsgcn/main.py \
    --stage det \
    --phase test \
    --config $config \
    --load_from $load_from \
    --save_output
