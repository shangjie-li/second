import argparse
import glob
from pathlib import Path
import time

try:
    import open3d
    from utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from data import cfg, cfg_from_yaml_file
from data import KittiDataset
from second_net import build_network, load_data_to_gpu
from utils import common_utils


class DemoDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, data_path=None, logger=None, ext='.bin'):
        """
        Args:
            dataset_cfg:
            class_names:
            training:
            data_path:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, logger=logger
        )
        self.data_path = data_path
        self.ext = ext
        file_list = glob.glob(str(data_path / f'*{self.ext}')) if self.data_path.is_dir() else [self.data_path]
        file_list.sort()
        self.sample_file_list = file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/config.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='data/kitti/training/velodyne/000008.bin',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='weights/second_7862.pth',
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of SECOND-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        data_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    print()
    print('<<< model >>>')
    print(model)
    print()
    
    """
    <<< model >>>
    SECONDNet(
      (vfe): MeanVFE()
      (backbone_3d): VoxelBackBone8x(
        (conv_input): SparseSequential(
          (0): SubMConv3d(4, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
          (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (conv1): SparseSequential(
          (0): SparseSequential(
            (0): SubMConv3d(16, 16, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv2): SparseSequential(
          (0): SparseSequential(
            (0): SparseConv3d(16, 32, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): SparseSequential(
            (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): SparseSequential(
            (0): SubMConv3d(32, 32, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv3): SparseSequential(
          (0): SparseSequential(
            (0): SparseConv3d(32, 64, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv4): SparseSequential(
          (0): SparseSequential(
            (0): SparseConv3d(64, 64, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[0, 1, 1], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (2): SparseSequential(
            (0): SubMConv3d(64, 64, kernel_size=[3, 3, 3], stride=[1, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
            (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
        (conv_out): SparseSequential(
          (0): SparseConv3d(64, 128, kernel_size=[3, 1, 1], stride=[2, 1, 1], padding=[0, 0, 0], dilation=[1, 1, 1], output_padding=[0, 0, 0], bias=False, algo=ConvAlgo.MaskImplicitGemm)
          (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (map_to_bev_module): HeightCompression()
      (backbone_2d): BaseBEVBackbone(
        (blocks): ModuleList(
          (0): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
            (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
          (1): Sequential(
            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
            (1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
            (2): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (3): ReLU()
            (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (6): ReLU()
            (7): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (8): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (9): ReLU()
            (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (11): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (12): ReLU()
            (13): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (14): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (15): ReLU()
            (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (17): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (18): ReLU()
          )
        )
        (deblocks): ModuleList(
          (0): Sequential(
            (0): ConvTranspose2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): Sequential(
            (0): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
      )
      (dense_head): AnchorHeadSingle(
        (cls_loss_func): SigmoidFocalClassificationLoss()
        (reg_loss_func): WeightedSmoothL1Loss()
        (dir_loss_func): WeightedCrossEntropyLoss()
        (conv_cls): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
        (conv_box): Conv2d(512, 42, kernel_size=(1, 1), stride=(1, 1))
        (conv_dir_cls): Conv2d(512, 12, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    """
    
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            
            print()
            print('<<< data_dict >>>')
            for key, val in data_dict.items():
                if isinstance(val, np.ndarray):
                    print(key, type(val), val.shape)
                    print(val)
                else:
                    print(key, type(val))
                    print(val)
            print()
            
            """
            <<< data_dict >>>
            points <class 'numpy.ndarray'> (60270, 5)
            [[ 0.    21.554  0.028  0.938  0.34 ]
             [ 0.    21.24   0.094  0.927  0.24 ]
             [ 0.    21.056  0.159  0.921  0.53 ]
             ...
             [ 0.     3.805 -1.433 -1.78   0.37 ]
             [ 0.     3.731 -1.391 -1.741  0.07 ]
             [ 0.     3.841 -1.419 -1.793  0.   ]]
            frame_id <class 'numpy.ndarray'> (1,)
            [0]
            voxels <class 'numpy.ndarray'> (35986, 5, 4)
            [[[21.554  0.028  0.938  0.34 ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]]
            
             [[21.24   0.094  0.927  0.24 ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]]
            
             [[21.056  0.159  0.921  0.53 ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]]
            
             ...
            
             [[ 3.793 -1.442 -1.776  0.36 ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]]
            
             [[ 3.805 -1.433 -1.78   0.37 ]
              [ 3.841 -1.419 -1.793  0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]]
            
             [[ 3.731 -1.391 -1.741  0.07 ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]
              [ 0.     0.     0.     0.   ]]]
            voxel_coords <class 'numpy.ndarray'> (35986, 4)
            [[  0  39 800 431]
             [  0  39 801 424]
             [  0  39 803 421]
             ...
             [  0  12 771  75]
             [  0  12 771  76]
             [  0  12 772  74]]
            voxel_num_points <class 'numpy.ndarray'> (35986,)
            [1 1 1 ... 1 2 1]
            batch_size <class 'int'>
            1
            """
            
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict) # 0
            
            time_start = time.time()
            pred_dicts, _ = model.forward(data_dict) # 1
            pred_dicts, _ = model.forward(data_dict) # 2
            pred_dicts, _ = model.forward(data_dict) # 3
            pred_dicts, _ = model.forward(data_dict) # 4
            pred_dicts, _ = model.forward(data_dict) # 5
            pred_dicts, _ = model.forward(data_dict) # 6
            pred_dicts, _ = model.forward(data_dict) # 7
            pred_dicts, _ = model.forward(data_dict) # 8
            pred_dicts, _ = model.forward(data_dict) # 9
            pred_dicts, _ = model.forward(data_dict) # 10
            time_end = time.time()
            
            print()
            print('<<< pred_dicts[0] >>>') # It seems that there is only one element in the list of pred_dicts.
            for key, val in pred_dicts[0].items():
                try:
                    print(key, type(val), val.shape)
                    print(val)
                except:
                    print(key, type(val))
                    print(val)
            print()
            
            """
            <<< pred_dicts[0] >>>
            pred_boxes <class 'torch.Tensor'> torch.Size([26, 7])
            tensor([[  6.5413,  -3.8676,  -1.0464,   3.1010,   1.4945,   1.4179,   5.9651],
                    [  4.0109,   2.6516,  -0.8394,   3.3870,   1.6194,   1.5364,   6.0269],
                    [ 14.7414,  -1.0834,  -0.7887,   3.7429,   1.5887,   1.5087,   5.9664],
                    [  8.1472,   1.2232,  -0.8027,   3.6676,   1.5530,   1.5924,   2.8546],
                    [ 33.6932,  -7.0879,  -0.4292,   4.3106,   1.8351,   1.8209,   2.8547],
                    [ 24.9487, -10.2688,  -0.9861,   3.8643,   1.6511,   1.4657,   5.8788],
                    [ 20.0299,  -8.3655,  -0.9303,   2.2176,   1.4840,   1.5422,   5.9772],
                    [ 28.7780,  -1.6511,  -0.3487,   3.7289,   1.5275,   1.5365,   4.4005],
                    [ 30.4293,  -3.7957,  -0.2656,   1.8893,   0.5680,   1.7781,   5.9896],
                    [ 55.6441, -20.2403,  -0.4769,   4.5618,   1.7342,   1.7406,   2.8706],
                    [ 40.9560,  -9.7300,  -0.5424,   3.8220,   1.6556,   1.6880,   5.9168],
                    [ 34.1309,  -4.9598,  -0.3965,   0.6571,   0.6801,   1.8507,   6.0455],
                    [ 18.6563,   0.2275,  -0.6291,   0.6412,   0.6515,   1.6988,   4.2951],
                    [ 33.5467, -15.3437,  -0.5308,   1.9204,   0.5285,   1.7438,   2.7655],
                    [  6.7900,  10.1602,  -0.6297,   0.5600,   0.5761,   1.6670,   2.7932],
                    [ 15.8195,   6.2329,  -0.3882,   0.4248,   0.5545,   1.7555,   4.7491],
                    [ 21.7721,   4.3337,  -0.1917,   0.2966,   0.4799,   1.7719,   4.7317],
                    [ 29.7343, -13.9505,  -0.8480,   1.6796,   0.5411,   1.6815,   2.8439],
                    [ 53.7386, -16.1984,  -0.3642,   1.8209,   0.5372,   1.7668,   2.8069],
                    [ 37.4135,  -6.1628,  -0.4630,   1.6738,   0.4957,   1.6488,   5.8689],
                    [ 14.5846, -13.8957,  -0.6122,   0.6816,   0.6205,   1.8725,   5.6815],
                    [  0.2837,  18.7304,  -0.9099,   3.9833,   1.5368,   1.4433,   6.3696],
                    [  2.1951,   6.6422,  -0.8483,   4.2948,   1.7002,   1.4900,   2.7774],
                    [ 40.5720,  -7.1177,  -0.4551,   0.6637,   0.6456,   1.8483,   6.0781],
                    [ 37.1179, -16.6012,  -0.7930,   1.6095,   0.5767,   1.6327,   2.8000],
                    [  0.9228,  -4.5643,  -0.8400,   0.3704,   0.5207,   1.8065,   1.3228]],
                   device='cuda:0')
            pred_scores <class 'torch.Tensor'> torch.Size([26])
            tensor([0.9858, 0.9799, 0.9705, 0.9467, 0.8207, 0.7406, 0.7404, 0.5991, 0.5088,
                    0.4350, 0.4235, 0.3794, 0.3011, 0.2578, 0.1976, 0.1885, 0.1748, 0.1658,
                    0.1509, 0.1469, 0.1437, 0.1418, 0.1288, 0.1154, 0.1088, 0.1016],
                   device='cuda:0')
            pred_labels <class 'torch.Tensor'> torch.Size([26])
            tensor([1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2, 2, 3, 2, 2, 2, 3, 3, 3, 2, 1, 1, 2,
                    3, 2], device='cuda:0')
            """

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)
                
            print('Time cost per batch: %s' % (round((time_end - time_start) / 10, 3)))

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
