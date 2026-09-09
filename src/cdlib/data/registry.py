"""DATASET_REGISTRY — all dataset classes register here.

Register your dataset with:
    @DATASET_REGISTRY.register("sysu_cd")
    class SYSUCDDataset(Dataset):
        def __getitem__(self, idx: int) -> dict:
            '''Return the frozen __getitem__ contract:
            {
                "img1": Tensor[C,H,W] float32 [0,1],
                "img2": Tensor[C,H,W] float32 [0,1],
                "mask": Tensor[1,H,W] float32 {0,1,-1},
                "nuisance_label": Tensor[] int64,
                "meta": {...},
            }
            '''
            ...
"""

from cdlib.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
