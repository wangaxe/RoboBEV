from .direct_with_lb import LowBoundedDIRECT_full_parrallel
import numpy as np
import torch
import copy

class FunctionalVerification():

    def __init__(self, 
                 model, 
                #  status, # UniAD.module
                 token:str,
                 frame:dict, 
                 ground_truth,
                 verify_loss): 
        self.model = model
        self.ori_frame = frame
        self.token = token
        self.images = self.get_img_tensor(frame)
        self.loss = verify_loss
        self.gt_cls = ground_truth
        self.nb_camera = self.images.shape[0]
        # self.status = status

    def update_frame(self, new_frame):
        self.ori_frame = new_frame
        self.images = self.get_img_tensor(new_frame)
    
    @staticmethod
    def _normlise_img(img_tensor:torch.Tensor) -> torch.Tensor:
        return ((img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()), (img_tensor.min(), img_tensor.max()))

    @staticmethod
    def _unnormlise_img(img_tensor:torch.Tensor, min_v, max_v) -> torch.Tensor:
        return img_tensor * (max_v - min_v) + min_v


    @staticmethod
    def get_img_tensor(mmcv_data:dict) -> torch.Tensor:
        return mmcv_data['img'][0].data[0].squeeze()

    @staticmethod
    def replace_img_tensor(ori_frame, perturb_imgs:torch.Tensor) -> dict:
        new_frame = copy.deepcopy(ori_frame)
        new_frame['img'][0].data[0].data = perturb_imgs.data
        return new_frame

    @staticmethod
    def functional_perturb(x, in_arr):
        raise NotImplementedError

    def verification(self, in_arrs):
        query_result = []
        for idx, in_arr in enumerate(in_arrs):
            # self.model.module = copy.deepcopy(self.status)
            perturbed_imgs = torch.zeros_like(self.images)
            in_arr = in_arr.reshape(self.nb_camera,-1)
            for i in range(self.nb_camera):
                perturbed_imgs[i] = self.functional_perturb(self.images[i], in_arr[i])
            tmp_frame = self.replace_img_tensor(self.ori_frame, perturbed_imgs.unsqueeze(0))
            with torch.no_grad():
                _pediction = self.model(return_loss=False, rescale=True, **tmp_frame)
            _query = self.loss(_pediction, self.token, self.gt_cls)
            query_result.append(_query)
        return np.array(query_result)

    def set_problem(self):
        return self.verification
