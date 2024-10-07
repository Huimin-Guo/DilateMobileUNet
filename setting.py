'''
Configs for training & testing
'''

import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    # data settings
    parser.add_argument(
        '--root_dir',
        default='./data',
        type=str,
        help='Root directory path of 2d data')

    parser.add_argument(
        '--cv5_name',
        default='ggo_cv5',
        type=str,
        help='the name of the cross validation name'
    )
    parser.add_argument(
        '--fold',
        default=0,
        type=int,
        help='the fold of the cross validation'
    )
    parser.add_argument(
        '--task_name',
        default="",
        type=str,
        help="if empty means trails, if not, it will be trailstask_name"
    )

    # model settings
    parser.add_argument(
        '--model_type',
        default="s",
        type=str,
        help="Model type: s, xs, xxs"
    )
    parser.add_argument(
        '--n_input_channels',
        default=1,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--n_seg_classes',
        default=3,
        type=int,
        help="Number of segmentation classes"
    )
    parser.add_argument(
        '--hidden_channels',
        default=128,
        type=int,
        help="Number of hidden channels for DeTrans"
    )
    parser.add_argument(
        '--model',
        default='acrunet',
        type=str,
        help='(resunet/resnext/acrunet')
    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    # training settings
    parser.add_argument(
        '--val',
        action='store_true',
        help='validation flag'
    )

    parser.add_argument(
        '--val_interval',
        default=2,
        type=int,
        help='Number of validation interval'
    )
    parser.add_argument(
        '--a_min',
        default=-1200,
        type=int,
        help='intensity original range min'
    )
    parser.add_argument(
        '--a_max',
        default=600,
        type=int,
        help='intensity original range max'
    )
    parser.add_argument(
        '--b_min',
        default=0.0,
        type=float,
        help='intensity target range min'
    )
    parser.add_argument(
        '--b_max',
        default=1.0,
        type=float,
        help='intensity target range max'
    )

    # refer to https://stackoverflow.com/a/33564487
    parser.add_argument(
        '--patch_size',
        nargs='+',
        type=int,
        help='the spatial size of the crop region for train, (h, w, d)',
        default=(256, 256)
    )
    parser.add_argument(
        '--patch_size_val',
        nargs='+',
        type=int,
        help='the spatial size of the crop region for val/test, (h, w, d)',
        default=(256, 256)
    )
    parser.add_argument(
        '--patch_overlap',
        default=0.5,
        type=float,
        help='Amount of overlap between scans.'
    )
    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=0.001,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--num_workers',
        default=0,
        type=int,
        help='Number of jobs')
    parser.add_argument(
        '--batch_size', default=2, type=int, help='Batch Size')
    parser.add_argument(
        '--phase', default='train', type=str, help='Phase of train/val/test')
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--save_intervals',
        default=50,
        type=int,
        help='Interation for saving model')

    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Path for resume model.'
    )
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='gpu id')

    args = parser.parse_args()
    args.save_folder = f"./trails{args.task_name}/fold_{args.fold}/{args.model}/{args.model_type}_{tuple(args.patch_size)[0]}"

    return args
