import pathlib
import statistics
import time
import argparse
import cv2
import kornia
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# from models.DenseNet_cat import DenseNet_half as Model
from models.DenseNet_add import DenseNet_half as Model

class Fuse:
    """
    fuse with infrared folder and visible folder
    """

    def __init__(self, model_path: str):
        """
        :param model_path: path of pre-trained parameters
        """

        # device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # model parameters
        params = torch.load(model_path, map_location='cpu')

        self.net = Model()

        self.net.load_state_dict(params['net'])

        self.net.to(device)
        
        self.net.eval()

    def __call__(self, i1_folder: str, i2_folder: str, dst: str, fuse_type = None):
        """
        fuse with i1 folder and vi folder and save fusion image into dst
        :param i1_folder: infrared image folder
        :param vi_folder: visible image folder
        :param dst: fusion image output folder
        """

        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        print('Model params: {:}'.format(para))

        # image list
        i1_folder = pathlib.Path(i1_folder)
        i2_folder = pathlib.Path(i2_folder)
        i1_list = sorted([x for x in sorted(i1_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])
        i2_list = sorted([x for x in sorted(i2_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])

        # check image name and fuse
        fuse_time = []
        rge = tqdm(zip(i1_list, i2_list))

        for i1_path, i2_path in rge:
            start = time.time()

            # check image name
            i1_name = i1_path.stem
            i2_name = i2_path.stem
            rge.set_description(f'fusing {i1_name}')
            # assert i1_name == vi_name

            # read image
            i1, i2 = self._imread(str(i1_path), str(i2_path), fuse_type = fuse_type)
            i1 = i1.unsqueeze(0).to(self.device)
            i2 = i2.unsqueeze(0).to(self.device)

            # network forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fu = self._forward(i1, i2)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


            # save fusion tensor
            fu_path = pathlib.Path(dst, i1_path.name)
            self._imsave(fu_path, fu)

            end = time.time()
            fuse_time.append(end - start)
        
        # time analysis
        if len(fuse_time) > 2:
            mean = statistics.mean(fuse_time[1:])
            print('fps (equivalence): {:.2f}'.format(1. / mean))

        else:
            print(f'fuse avg time: {fuse_time[0]:.2f}')


    @torch.no_grad()
    def _forward(self, i1: torch.Tensor, i2: torch.Tensor) -> torch.Tensor:
        fusion = self.net(i1, i2)
        return fusion

    @staticmethod
    def _imread(i1_path: str, i2_path: str, flags=cv2.IMREAD_GRAYSCALE, fuse_type = None) -> torch.Tensor:
        i1_cv = cv2.imread(i1_path, flags).astype('float32')
        
        i2_cv = cv2.imread(i2_path, flags).astype('float32')
        height, width = i1_cv.shape[:2]
        # if fuse_type == 'black_ir':
        #     i1_cv[True] = 0
        # elif fuse_type == 'black_vi':
        #     i2_cv[True] = 0
        # if fuse_type == 'white_ir':
        #     i1_cv[True] = 255
        # if fuse_type == 'white_vi':
        #     i2_cv[True] = 255
            
        i1_ts = kornia.utils.image_to_tensor(i1_cv / 255.0).type(torch.FloatTensor)
        i2_ts = kornia.utils.image_to_tensor(i2_cv / 255.0).type(torch.FloatTensor)
        return i1_ts, i2_ts

    @staticmethod
    def _imsave(path: pathlib.Path, image: torch.Tensor):
        im_ts = image.squeeze().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(path), im_cv)



if __name__ == '__main__':
    model = 'densenet_add_half'
    f = Fuse(f"./cache/{model}/best.pth")

    parser = argparse.ArgumentParser()
    parser.add_argument("--i1", default='../datasets/Multi_spectral/infrared/test', help="i1 path")
    parser.add_argument("--i2", default='../datasets/Multi_spectral/visible/test', help="i2 path")
    args = parser.parse_args()
    
    # f(args.i1, args.i2, f'runs/test/{model}/black_ir', 'black_ir')
    # f(args.i1, args.i2, f'runs/test/{model}/black_vi', 'black_vi')
    # f(args.i1, args.i2, f'runs/test/{model}/white_ir', 'white_ir')
    # f(args.i1, args.i2, f'runs/test/{model}/white_vi', 'white_vi')
    f(args.i1, args.i2, f'runs/Multi_spectral/{model}/fuse')
