config=./dsgcn/configs/cfg_test_det_ms1m_8_prpsls.py
load_from=./data/pretrained_models/pretrained_gcn_d_iop_ms1m.pth

export CUDA_VISIBLE_DEVICES=7

PYTHONPATH=. python dsgcn/main.py \
    --det_label 'iop' \
    --stage det \
    --phase test \
    --config $config \
    --load_from $load_from \
    --save_output
