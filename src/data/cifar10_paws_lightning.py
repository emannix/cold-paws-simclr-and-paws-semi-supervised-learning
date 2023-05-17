
import torchvision.transforms as transforms
import torchvision
import numpy as np
from PIL import Image
from pdb import set_trace as pb

def init_transform(targets, samples):
    # return np.array(samples), np.array(targets)
    return targets, samples

class TransCIFAR10(torchvision.datasets.CIFAR10):

    def __init__(
        self,
        root,
        image_folder='cifar-pytorch/11222017/',
        tar_file='cifar-10-python.tar.gz',
        copy_data=False,
        train=True,
        transform=None,
        target_transform=None,
        multicrop_transform=(0, None),
        supervised_views=1,
        download=True
    ):
        # data_path = None
        # if copy_data:
        #     logger.info('copying data locally')
        #     data_path = copy_cifar10_locally(
        #         root=root,
        #         image_folder=image_folder,
        #         tar_file=tar_file)
        # if (not copy_data) or (data_path is None):
        #     data_path = os.path.join(root, image_folder)
        data_path = root

        super().__init__(data_path, train, transform, target_transform, download)

        self.supervised_views = supervised_views
        self.multicrop_transform = multicrop_transform
        self.init_transform = init_transform

    def set_supervised(self, labelled_indices):
        self.targets[labelled_indices], self.data[labelled_indices] = self.init_transform(self.targets[labelled_indices], self.data[labelled_indices])
        mint = None
        self.target_indices = []
        effective_classes = 0
        for t in range(len(self.classes)):
            indices = np.squeeze(np.argwhere(self.targets[labelled_indices] == t)).tolist()
            if not isinstance(indices, list): # this takes care of instances when only one index for a class. It will still fail if there are not single class instances
                indices = [indices]
            if len(indices) > 0:
                # indices = labelled_indices[indices]
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                effective_classes += 1
        return effective_classes

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:

            if self.supervised:
                return *[self.transform(img) for _ in range(self.supervised_views)], target

            else:
                img_1 = self.transform(img)
                img_2 = self.transform(img)

                multicrop, mc_transform = self.multicrop_transform
                if multicrop > 0 and mc_transform is not None:
                    mc_imgs = [mc_transform(img) for _ in range(int(multicrop))]
                    return img_1, img_2, *mc_imgs, target

                return img_1, img_2, target

        return img, target