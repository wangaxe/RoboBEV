# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
os.chdir('/home/fu/workspace/RoboBEV/zoo/DETR3D')
import sys
sys.path.append("/home/fu/workspace/RoboBEV/zoo/DETR3D")
import copy
import time
import warnings

import numpy as np
import torch

import mmcv
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from projects.mmdet3d_plugin.datasets import CustomNuScenesDataset
from projects.mmdet3d_plugin.verification import (FunctionalVerification,
     LowBoundedDIRECT_full_parrallel)
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from kornia.enhance import (adjust_brightness, adjust_gamma,
                            adjust_hue, adjust_saturation)
from torchvision.utils import save_image


class OpticalVerification(FunctionalVerification):
    @staticmethod
    def functional_perturb(x, in_arr):
        transformed = torch.zeros_like(x)
        # hue, saturation, contrast, bright = \
        #     in_arr[0], in_arr[1], in_arr[2], in_arr[3]
        # hue, saturation, contrast = in_arr[0], in_arr[1], in_arr[2]
        x, min_max_v = OpticalVerification._normlise_img(x)
        hue, saturation = in_arr[0], in_arr[1]
        with torch.no_grad():
            x = adjust_hue(x, hue)
            x = adjust_saturation(x, saturation)
            # x = adjust_gamma(x, contrast)
            # x = adjust_brightness(x, bright)
        x = OpticalVerification._unnormlise_img(x, *min_max_v)
        transformed = x
        return transformed 

def get_args(arg_lst):
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')

    # perturbation
    parser.add_argument('--hue', default=0.1, type=float,
        help='range should be in [-PI, PI], while 0 means no shift'
        'control: [ - defautl * PI, defautl * PI]')
    parser.add_argument('--saturation', default=0.2, type=float,
        help='range should be in [0, 2], while 1 means no shift'
        'control: [ 1 - defautl, 1 + defautl]')
    parser.add_argument('--contrast', default=0.0, type=float,
        help='range should be in [0, 2], while 1 means no shift'
        'control: [ 1 - defautl, 1 + defautl]')
    parser.add_argument('--bright', default=0.0, type=float,
        help='range should be in [0, 1], while 0 means no shift'
        'control: TBD')
    # DIRECT
    parser.add_argument('--max-evaluation', default=500, type=int)
    parser.add_argument('--max-deep', default=10, type=int)
    parser.add_argument('--po-set', action='store_true')
    parser.add_argument('--po-set-size', default=1, type=int)
    parser.add_argument('--max-iteration', default=50, type=int)
    parser.add_argument('--tolerance', default=1e-5, type=float)

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(arg_lst)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


simplied_class_names = ['car', 'truck', 'bus', 
                        'pedestrian', 'motorcycle', 
                        'bicycle', 'traffic_cone']

def main():
    # args = parse_args()
    arg_lst = [
        '/home/fu/workspace/RoboBEV/zoo/DETR3D/projects/configs/detr3d/detr3d_res101_gridmask.py',
        '/datasets/bev_models/detr3d_resnet101.pth',
        '--eval', 'bbox',
    ]
    args = get_args(arg_lst)
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    # dataset = build_dataset(cfg.data.test)
    data_arg = cfg.data.test.copy()
    data_arg.pop('type')
    dataset = CustomNuScenesDataset(**data_arg)
    # print(dataset.eval_detection_configs)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

###################################################
#               VERIFICATION
###################################################
    bound = []
    if args.hue != 0:
        bound.append([-np.pi*args.hue, np.pi*args.hue])
    if args.saturation != 0:
        bound.append([1-args.saturation, 1+args.saturation])
    if args.contrast != 0:
        bound.append([1-args.contrast, 1+args.contrast])
    assert len(bound) != 0
    bound = bound*6
    print(bound)

    kwargs = {} if args.eval_options is None else args.eval_options
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=args.eval, **kwargs))
    # print(eval_kwargs)

    # gt_boxes = get_gt_boxes(dataset)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    query_model = copy.deepcopy(model)

    for i, data in enumerate(data_loader):
        # print(data['img'][0].data[0].max())
        # print(data['img'][0].data[0].min())
        token = dataset.data_infos[i]['token']
        task = OpticalVerification(query_model,
                            # copy.deepcopy(model.module),
                            token,
                            data,
                            simplied_class_names,
                            dataset.results2dist)
        object_func = task.set_problem()
        direct_solver = LowBoundedDIRECT_full_parrallel(object_func,len(bound),
                                                bound, 
                                                args.max_iteration, 
                                                args.max_deep, 
                                                args.max_evaluation, 
                                                args.tolerance,
                                                debug=False)
        start_time = time.time()
        direct_solver.solve()
        end_time = time.time()
        print(f"{i+1}-th frame, time: {(end_time-start_time)/60:.2f} min")

        print(list(direct_solver.optimal_result()))
        # optimal = np.reshape(np.array([0.0, 1.1333333333333333, 0.0, 1.1333333333333333, -0.20943951023931953, 1.1333333333333333, -0.2792526803190927, 1.0, -0.20943951023931953, 1.0, -0.20943951023931953, 1.0]), (-1,2))
        # print(optimal)
        optimal = np.reshape(direct_solver.optimal_result(), (-1,2))
        ori_img = OpticalVerification.get_img_tensor(data)
        tmp_img = torch.zeros_like(ori_img)
        for i in range(ori_img.shape[0]):
            # save_image(OpticalVerification._normlise_img(ori_img[i])[0], f'./ori_img_{i}.png')
            tmp_img[i] = OpticalVerification.functional_perturb(ori_img[i],optimal[i])
            # save_image(OpticalVerification._normlise_img(tmp_img[i])[0], f'./tmp_img_{i}.png')

        tmp_data = OpticalVerification.replace_img_tensor(copy.deepcopy(data), tmp_img.unsqueeze(0))

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **tmp_data)
        dist = dataset.results2dist(result, token, simplied_class_names)
        print(f"distance: {dist}")

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        dist = dataset.results2dist(result, token, simplied_class_names)
        print(f"distance: {dist}")

        break


if __name__ == '__main__':
    main()
