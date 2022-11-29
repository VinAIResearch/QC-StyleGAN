import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from utils import data_utils


transform_deg = A.Compose(
    [
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(5, 31), p=1.0),
                A.MotionBlur(blur_limit=(5, 31), p=1.0),
                A.MedianBlur(blur_limit=(5, 31), p=1.0),
                A.Blur(blur_limit=(5, 25), p=1.0),
            ],
            p=0.8,
        ),
        A.OneOf(
            [
                A.GaussNoise(var_limit=(200.0, 650.0), p=1.0),
            ],
            p=0.8,
        ),
        A.OneOf(
            [
                A.Downscale(scale_min=0.2, scale_max=0.9, p=1.0),
                A.ImageCompression(quality_lower=30, quality_upper=50, p=1.0),
            ],
            p=0.8,
        ),
    ]
)

transform = A.Compose(
    [
        A.geometric.resize.Resize(256, 256, cv2.INTER_CUBIC),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ],
    additional_targets={"image2": "image"},
)


class ImagesDataset(Dataset):
    def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
        self.source_paths = sorted(data_utils.make_dataset(source_root))
        self.target_paths = sorted(data_utils.make_dataset(target_root))
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = cv2.cvtColor(cv2.imread(from_path), cv2.COLOR_BGR2RGB)
        to_im = transform_deg(image=from_im)["image"]

        transformed = transform(image=from_im, image2=to_im)

        from_im, to_im = transformed["image"], transformed["image2"]
        # from_im = Image.open(from_path)
        # from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

        # to_path = self.target_paths[index]
        # to_im = Image.open(to_path).convert('RGB')
        # if self.target_transform:
        #     to_im = self.target_transform(to_im)

        # if self.source_transform:
        #     from_im = self.source_transform(from_im)
        # else:
        #     from_im = to_im

        return from_im, to_im
