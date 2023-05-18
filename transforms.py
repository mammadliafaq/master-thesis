import albumentations
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config):
    return albumentations.Compose(
        [
            albumentations.Resize(
                config.dataset.input_size, config.dataset.input_size, always_apply=True
            ),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            # albumentations.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
            # albumentations.ShiftScaleRotate(
            #  shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            # ),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms(config):
    return albumentations.Compose(
        [
            albumentations.Resize(
                config.dataset.input_size, config.dataset.input_size, always_apply=True
            ),
            albumentations.Normalize(),
            ToTensorV2(p=1.0),
        ]
    )
