dataset_paths = {
    # Human Faces
    "ffhq_deg_train_source": "../ffhq_deg/train/sharp",
    "ffhq_deg_train_target": "../ffhq_deg/train/deg",
    "ffhq_deg_test_source": "../ffhq_deg/test/sharp",
    "ffhq_deg_test_target": "../ffhq_deg/test/deg",
    # AFHQ
    "afhq_cat_train_source": "../datasets/afhq/train/cat",
    "afhq_cat_train_target": "../datasets/afhq/train/cat",
    "afhq_cat_test_source": "../datasets/afhq/val/cat",
    "afhq_cat_test_target": "../datasets/afhq/val/cat",
    # Church
    "church_train_source": "../church_256_train",
    "church_train_target": "../church_256_train",
    "church_test_source": "../church_256_val",
    "church_test_target": "../church_256_val",
}

model_paths = {
    # 'stylegan2_ada_ffhq': 'pretrained_models/stylegan2-church-torch.pkl',
    # 'stylegan2_ada_ffhq': 'pretrained_models/strong_aug_wide_range.pkl',
    # 'stylegan2_ada_ffhq': 'pretrained_models/network-snapshot-cat.pkl',
    "stylegan2_ada_ffhq": "",
    "ir_se50": "pretrained_models/model_ir_se50.pth",
    "circular_face": "pretrained_models/CurricularFace_Backbone.pth",
    "mtcnn_pnet": "pretrained_models/mtcnn/pnet.npy",
    "mtcnn_rnet": "pretrained_models/mtcnn/rnet.npy",
    "mtcnn_onet": "pretrained_models/mtcnn/onet.npy",
    "shape_predictor": "shape_predictor_68_face_landmarks.dat",
    "moco": "pretrained_models/moco_v2_800ep_pretrain.pt",
    "resnet34": "pretrained_models/resnet34-333f7ec4.pth",
}
