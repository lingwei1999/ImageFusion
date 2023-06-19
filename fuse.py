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

    def __call__(self, ir_folder: str, vi_folder: str, dst: str, fuse_type = None):
        """
        fuse with ir folder and vi folder and save fusion image into dst
        :param ir_folder: infrared image folder
        :param vi_folder: visible image folder
        :param dst: fusion image output folder
        """

        para = sum([np.prod(list(p.size())) for p in self.net.parameters()])
        print('Model params: {:}'.format(para))

        # image list
        ir_folder = pathlib.Path(ir_folder)
        vi_folder = pathlib.Path(vi_folder)
        ir_list = sorted([x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])
        vi_list = sorted([x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.bmp', '.png', '.jpg', '.JPG']])

        # check image name and fuse
        fuse_time = []
        rge = tqdm(zip(ir_list, vi_list))

        for ir_path, vi_path in rge:
            start = time.time()

            # check image name
            ir_name = ir_path.stem
            vi_name = vi_path.stem
            rge.set_description(f'fusing {ir_name}')
            # assert ir_name == vi_name

            # read image
            ir, vi = self._imread(str(ir_path), str(vi_path), fuse_type = fuse_type)
            ir = ir.unsqueeze(0).to(self.device)
            vi = vi.unsqueeze(0).to(self.device)

            # network forward
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            fu = self._forward(ir, vi)
            torch.cuda.synchronize() if torch.cuda.is_available() else None


            # save fusion tensor
            fu_path = pathlib.Path(dst, ir_path.name)
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
    def _forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        fusion = self.net(ir, vi)
        return fusion

    @staticmethod
    def _imread(ir_path: str, vi_path: str, flags=cv2.IMREAD_GRAYSCALE, fuse_type = None) -> torch.Tensor:
        ir_cv = cv2.imread(ir_path, flags).astype('float32')
        
        vi_cv = cv2.imread(vi_path, flags).astype('float32')
        height, width = ir_cv.shape[:2]
            
        ir_ts = kornia.utils.image_to_tensor(ir_cv / 255.0).type(torch.FloatTensor)
        vi_ts = kornia.utils.image_to_tensor(vi_cv / 255.0).type(torch.FloatTensor)
        return ir_ts, vi_ts

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
    parser.add_argument("--ir", default='../datasets/test/ir', help="ir path")
    parser.add_argument("--vi", default='../datasets/test/vi', help="vi path")
    args = parser.parse_args()
    
    f(args.ir, args.vi, f'runs/test/{model}/fuse')
