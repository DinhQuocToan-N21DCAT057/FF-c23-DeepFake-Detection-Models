import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs import Configs

cfg = Configs()

def get_transform(model_key: str):
    """Trả về Albumentations transform theo model_key"""
    mdl_cfg = cfg.model_cfgs.get(model_key)
    if mdl_cfg is None:
        raise ValueError(f"Invalid model_key: {model_key}")

    return A.Compose([
        A.Resize(mdl_cfg["size"], mdl_cfg["size"]),
        A.CenterCrop(mdl_cfg["crop"], mdl_cfg["crop"]),
        A.Normalize(mean=mdl_cfg["mean"], std=mdl_cfg["std"]),
        ToTensorV2(),
    ])