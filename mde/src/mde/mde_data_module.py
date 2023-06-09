import pathlib
from typing import Optional, Dict, Union, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from link_bot_data.new_dataset_utils import fetch_mde_dataset
from link_bot_data.wandb_datasets import get_dataset_with_version
from mde.torch_mde_dataset import TorchMDEDataset
from moonshine.filepath_tools import load_params
from moonshine.moonshine_utils import get_num_workers
from moonshine.torch_datasets_utils import my_collate, dataset_take, dataset_skip, dataset_repeat
from state_space_dynamics.torch_dynamics_dataset import remove_keys


class MDEDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_dir: Union[pathlib.Path, List[pathlib.Path]],
                 batch_size: int,
                 take: int,
                 skip: int,
                 repeat: Optional[int] = None,
                 val_dataset_dir: Optional[pathlib.Path] = None,
                 train_mode: str = 'train',
                 val_mode: str = 'val',
                 test_mode: str = 'test'):
        super().__init__()
        self.dataset_dir = dataset_dir
        if val_dataset_dir is not None:
            self.val_dataset_dir = val_dataset_dir
        else:
            self.val_dataset_dir = dataset_dir
        self.fetched_dataset_dir = None
        self.batch_size = batch_size
        self.take = take
        self.skip = skip
        self.repeat = repeat
        self.train_mode = train_mode
        self.val_mode = val_mode
        self.test_mode = test_mode

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # NOTE: I'm not using prepare_data or setup correctly here. This is because in order to write the relevant info
        #  in `self.add_dataset_params` I need to have actually constructed the datasets
        if isinstance(self.dataset_dir, list):
            self.fetched_dataset_dir = [fetch_mde_dataset(d) for d in self.dataset_dir]
            self.val_fetched_dataset_dir = [fetch_mde_dataset(d) for d in self.dataset_dir]
            self.dataset_hparams = load_params(self.fetched_dataset_dir[-1])
        else:
            self.fetched_dataset_dir = fetch_mde_dataset(self.dataset_dir)
            self.val_fetched_dataset_dir = fetch_mde_dataset(self.val_dataset_dir)
            self.dataset_hparams = load_params(self.fetched_dataset_dir)

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([remove_keys("scene_msg", "sdf", "sdf_grad")])
        train_dataset = TorchMDEDataset(self.fetched_dataset_dir, mode=self.train_mode, transform=transform)
        train_dataset_take = dataset_take(train_dataset, self.take)
        train_dataset_skip = dataset_skip(train_dataset_take, self.skip)

        self.train_dataset = dataset_repeat(train_dataset_skip, self.repeat)

        print("Applying take to validation & test as well")
        if self.val_dataset_dir is None:
            val_dataset = TorchMDEDataset(self.fetched_dataset_dir, mode=self.val_mode, transform=transform)
        else:
            val_dataset = TorchMDEDataset(self.val_fetched_dataset_dir, mode=self.train_mode, transform=transform)
        val_dataset_take = dataset_take(val_dataset, self.take)
        self.val_dataset = val_dataset_take

        test_dataset = TorchMDEDataset(self.fetched_dataset_dir, mode=self.test_mode, transform=transform)
        test_dataset_take = dataset_take(test_dataset, self.take)
        self.test_dataset = test_dataset_take
        print("NUM WORKERS = 0")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=my_collate,
                          num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=my_collate,
                          num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=my_collate,
                          num_workers=0)

    def add_dataset_params(self, model_params: Dict):
        model_params['take'] = self.take
        model_params['batch_size'] = self.batch_size
        model_params['skip'] = self.skip
        model_params['repeat'] = self.repeat
        model_params['dataset_dir'] = self.fetched_dataset_dir
        model_params['dataset_dir_versioned'] = get_dataset_with_version(self.dataset_dir, 'mde')
        model_params['dataset_hparams'] = self.dataset_hparams
        model_params['scenario'] = self.dataset_hparams['scenario']
