import os
from typing import ValuesView
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class ConditionalDataset(BaseDataset):
    """
    BASED ON UnalignedDataset
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir = os.path.join(opt.dataroot, opt.phase)  # create a path '/path/to/data/train'
        self.list_dataset_imgs()

        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        
        input_nc = self.opt.input_nc + self.nb_colors
        output_nc = self.opt.output_nc + self.nb_colors
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        
        # Zelfgeschreven, niet van toepassing voor training
        # self.input_color, self.output_color = self.opt.direction.strip().split(",")
        # assert self.input_color in self.colormap.keys()
        # assert self.output_color in self.colormap.keys()


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        # A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        # B_path = self.B_paths[index_B]
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        # # apply image transformation
        # A = self.transform_A(A_img)
        # B = self.transform_B(B_img)

        # choose 2 random different colors
        color_set = self.colors_train if self.opt.phase == 'train' else self.colors_test
        color_A, color_B = np.random.choice(color_set, size=2, replace=False)

        A_path = self.datasets[color_A][index % self.sizes[color_A]]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.sizes[color_B]
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.sizes[color_B] - 1)
        B_path = self.datasets[color_B][index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'color_A': self.colormap[color_A], 'color_B': self.colormap[color_B]}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        # return max(self.A_size, self.B_size)
        
        return max(*self.sizes.values())

    # def parse_dataset_file(self, path):
    #     """
    #         example json format:
    #         {
    #             "colors" : ["black","white","yellow","green"],
    #             "train_imgs" : [
    #                 {
    #                     "img":"img_train_1_input.png",
    #                     "color":"black"
    #                 },
    #                 {
    #                     "img":"img_train_2_input.png",
    #                     "color":"yellow"
    #                 }
    #             ],
    #             "test_imgs" : [
    #                 {
    #                     "img":"img_train_3_input.png",
    #                     "color":"green"
    #                 }
    #             ]
    #         }
        
    #     """
    #     if not os.path.isfile(path):
    #         raise ValueError(f"Provided dataset file '{path}' is not a file.")

    #     with open(path, 'r') as f:
    #         dataset_content = json.load(f)
        
    #     # determine number of different colors as stated in the json dataset file
    #     self.colormap = {color:i for i,color in enumerate(dataset_content['colors'])}
    #     self.nb_colors = len(self.colormap.keys())

    #     # store images per color in dictionary
    #     self.datasets_train = {i:[] for i in range(self.nb_colors)}
    #     self.datasets_test = {i:[] for i in range(self.nb_colors)}

    #     for dataset_key, dataset in [('train_imgs', self.datasets_train), ('test_imgs', self.datasets_test), ]:
    #         for img in dataset_content[dataset_key]:
    #             if not img['color'] in self.colormap.keys():
    #                 print(f"Error loading dataset: invalid color '{img['color']}' for image '{img['img']}'.")
    #                 continue
    #             dataset[self.colormap[img['color']]].append(img['img'])

    def get_colors_list(self, phases=None):
        colors = []
        for phase in (os.listdir(self.opt.dataroot) if phases == None else phases):
            current_dir = os.path.join(self.opt.dataroot, phase)
            colors += [color for color in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, color))]
        return sorted(list(set(colors)))

    def get_train_colors_list(self):
        return self.get_colors_list(phases=['train'])
    
    def get_test_colors_list(self):
        return self.get_colors_list(phases=['test'])

    def list_dataset_imgs(self):
        colors = self.get_colors_list()
        self.colormap = {color:i for i,color in enumerate(colors)}
        self.nb_colors = len(self.colormap.keys())

        self.datasets = {color:[] for color in self.colormap.keys()}
        self.sizes = {color:0 for color in self.colormap.keys()}
        
        self.colors_train = self.get_train_colors_list()
        self.colors_test = self.get_test_colors_list()

        for color in self.colormap.keys():
            if os.path.exists(os.path.join(self.dir, color)):
                self.datasets[color] = sorted(make_dataset(os.path.join(self.dir,color), self.opt.max_dataset_size))
                self.sizes[color] = len(self.datasets[color])
        