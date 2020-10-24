import re
import os
import os.path as osp
import torchvision.datasets as datasets
from torchvision.datasets.folder import has_file_allowed_extension, default_loader, IMG_EXTENSIONS


def make_dataset_with_annopath(directory,  annotation_path, extensions, is_valid_file=None):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    with open(annotation_path, "r") as f:
        contents = f.readlines()
        for line_str in contents:
            path_contents = [c.strip('/\r\n') for c in
                             re.split('\t', line_str)]
            assert len(path_contents) == 2
            img_file_name = path_contents[0]
            class_index = int(path_contents[1])
            path = os.path.join(directory, img_file_name)
            if is_valid_file(path):
                item = path, class_index
                instances.append(item)

    return instances


class DatasetFolder(datasets.DatasetFolder):
    def __init__(self, root, anno_file, loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None):
        super(datasets.DatasetFolder, self).__init__(root, transform=transform,
                                                     target_transform=target_transform)
        self.anno_file = anno_file
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset_with_annopath(self.root, anno_file, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = []
        class_to_idx = {}
        with open(self.anno_file, "r") as f:
            contents = f.readlines()
            for line_str in contents:
                path_contents = [c.strip('/\r\n') for c in
                                 re.split('\t', line_str)]
                assert len(path_contents) == 2
                img_file_name = path_contents[0]
                class_index = int(path_contents[1])
                class_name = osp.dirname(img_file_name)
                classes.append(class_name)
                class_to_idx[class_name] = class_index
        return classes, class_to_idx


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, anno_file, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(ImageFolder, self).__init__(root, anno_file, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
