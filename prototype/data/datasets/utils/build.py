from .registry import Registry
from difflib import SequenceMatcher
import os
import json
import errno
import os.path as osp
from PIL import Image

DATASET_REGISTRY = Registry("DATASET")


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError("No file exists at {}".format(path))

    while True:
        try:
            img = Image.open(path).convert("RGB")
            return img
        except IOError:
            print(
                "Cannot read image from {}, "
                "probably due to heavy IO. Will re-try".format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, "w") as f:
        json.dump(obj, f, indent=4, separators=(",", ": "))


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def get_most_similar_str_to_a_from_b(a, b):
    """Return the most similar string to a in b.

    Args:
        a (str): probe string.
        b (list): a list of candidate strings.
    """
    highest_sim = 0
    chosen = None
    for candidate in b:
        sim = SequenceMatcher(None, a, candidate).ratio()
        if sim >= highest_sim:
            highest_sim = sim
            chosen = candidate
    return chosen


def check_availability(requested, available):
    """Check if an element is available in a list.

    Args:
        requested (str): probe string.
        available (list): a list of available strings.
    """
    if requested not in available:
        psb_ans = get_most_similar_str_to_a_from_b(requested, available)
        raise ValueError(
            "The requested one is expected "
            "to belong to {}, but got [{}] "
            "(do you mean [{}]?)".format(available, requested, psb_ans)
        )


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
