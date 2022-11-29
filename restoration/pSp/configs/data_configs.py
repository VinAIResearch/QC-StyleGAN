from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
    "ffhq_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["ffhq_deg_train_source"],
        "train_target_root": dataset_paths["ffhq_deg_train_target"],
        "test_source_root": dataset_paths["ffhq_deg_test_source"],
        "test_target_root": dataset_paths["ffhq_deg_test_target"],
    },
    "afhq_cat_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["afhq_cat_train_source"],
        "train_target_root": dataset_paths["afhq_cat_train_target"],
        "test_source_root": dataset_paths["afhq_cat_test_source"],
        "test_target_root": dataset_paths["afhq_cat_test_target"],
    },
    "church_encode": {
        "transforms": transforms_config.EncodeTransforms,
        "train_source_root": dataset_paths["church_train_source"],
        "train_target_root": dataset_paths["church_train_target"],
        "test_source_root": dataset_paths["church_test_source"],
        "test_target_root": dataset_paths["church_test_target"],
    },
}
