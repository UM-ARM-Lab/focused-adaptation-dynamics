import pathlib

from link_bot_data.dataset_constants import DEFAULT_VAL_SPLIT, DEFAULT_TEST_SPLIT
from link_bot_data.files_dataset import OldDatasetSplitter
from link_bot_data.load_dataset import guess_dataset_format


def split_dataset(dataset_dir: pathlib.Path,
                  val_split=DEFAULT_VAL_SPLIT,
                  test_split=DEFAULT_TEST_SPLIT):
    dataset_format = guess_dataset_format(dataset_dir)
    if dataset_format == 'tfrecord':
        files_dataset = OldDatasetSplitter(root_dir=dataset_dir)
        sorted_records = sorted(list(dataset_dir.glob(f"example_*.tfrecords")))
        for file in sorted_records:
            files_dataset.add(file)
        files_dataset.split()
    elif dataset_format == 'pkl':
        split_dataset_via_files(dataset_dir, 'pkl', val_split, test_split)


def split_dataset_via_files(dataset_dir: pathlib.Path,
                            extension: str,
                            val_split=DEFAULT_VAL_SPLIT,
                            test_split=DEFAULT_TEST_SPLIT):
    paths = sorted(list(dataset_dir.glob(f"example_*.{extension}")))
    n_files = len(paths)
    n_validation = int(val_split * n_files)
    n_testing = int(test_split * n_files)
    val_files = paths[0:n_validation]
    paths = paths[n_validation:]
    test_files = paths[0:n_testing]
    train_files = paths[n_testing:]

    write_mode(dataset_dir, train_files, 'train')
    write_mode(dataset_dir, test_files, 'test')
    write_mode(dataset_dir, val_files, 'val')


def write_mode(dataset_dir, filenames, mode):
    with (dataset_dir / f"{mode}.txt").open("w") as f:
        for _filename in filenames:
            f.write(_filename.name + '\n')
