from os import path as osp

import numpy as np
# import mmcv
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist,\
    filter_eval_boxes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, \
    DetectionMetrics, DetectionBox, DetectionMetricDataList
from nuscenes.eval.common.utils import center_distance

from .data_utils import output_to_nusc_box, lidar_nusc_box_to_global


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.eval_set = eval_set_map[self.version]
        self.gt_boxes = self.get_gt_boxes()
        # tmp_dir = tempfile.TemporaryDirectory()
        # self.jsonfile_prefix = osp.join(tmp_dir.name, 'results')

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict
    
    def process_nusc_boxes(self, boxes):
        class_range = self.eval_detection_configs.class_range 
        boxes = add_center_dist(self.nusc, boxes)
        boxes = filter_eval_boxes(self.nusc, boxes, class_range, verbose=False)
        return boxes

    def get_gt_boxes(self, eval_set='mini_val', verbose=False):
        class_range = self.eval_detection_configs.class_range 
        gt_boxes = load_gt(self.nusc, 'mini_val', DetectionBox, verbose=False)
        gt_boxes = self.process_nusc_boxes(gt_boxes)
        return gt_boxes

    def results2dist(self,
                    result: dict,
                    # gt_boxes,
                    token,
                    class_names,
                    negative=True,
                    verbose=False
                    ):
        mapped_class_names = self.CLASSES
        mean_dist = 0
        nb_positive_cls = 0
        results_ = result[0]['pts_bbox']
        # results_ = [out['pts_bbox'] for out in result]
        # pred_boxes = self.format_bbox(results_)
        boxes = output_to_nusc_box(results_)
        boxes = lidar_nusc_box_to_global(
                                    self.data_infos[0], boxes,
                                    mapped_class_names,
                                    self.eval_detection_configs,
                                    self.eval_version)
        pred_boxes = get_pred_boxes_single_frame(boxes, 
                                                 token, 
                                                 mapped_class_names)
        pred_boxes = self.process_nusc_boxes(pred_boxes)

        for class_name in class_names:
            class_distance = 0
            for pred_box in pred_boxes.all:
                if pred_box.detection_name != class_name: continue
                for gt_idx, gt_box in enumerate(self.gt_boxes[pred_box.sample_token]):
                    if gt_box.detection_name == class_name:
                        class_distance += center_distance(gt_box, pred_box)
            if class_distance > 0:
                nb_positive_cls += 1
                mean_dist += class_distance
                # if verbose: print(class_name,' - ', class_distance)
        if nb_positive_cls != 0:
            if negative: return -1 * mean_dist/nb_positive_cls
            else: return mean_dist/nb_positive_cls
        else: return 0

def get_pred_boxes_single_frame(nusc_box,
                                sample_token,
                                mapped_class_names):
    nusc_annos = {}
    annos = []
    for i, box in enumerate(nusc_box):
        name = mapped_class_names[box.label]
        if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
            if name in [
                    'car',
                    'construction_vehicle',
                    'bus',
                    'truck',
                    'trailer',
            ]:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = NuScenesDataset.DefaultAttribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = NuScenesDataset.DefaultAttribute[name]

        nusc_anno = dict(
            sample_token=sample_token,
            translation=box.center.tolist(),
            size=box.wlh.tolist(),
            rotation=box.orientation.elements.tolist(),
            velocity=box.velocity[:2].tolist(),
            detection_name=name,
            detection_score=box.score,
            attribute_name=attr,
        )
        annos.append(nusc_anno)
        nusc_annos[sample_token] = annos
    return EvalBoxes.deserialize(nusc_annos, DetectionBox)