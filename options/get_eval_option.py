from argparse import Namespace
import re
from os.path import join as pjoin


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'finest'

    if opt.dataset_name == 'aistpp':
        opt.data_root = './dataset/AIST++_dataset'
        opt.motion_dir = pjoin(opt.data_root, 'annotations/motions') # using smpl 72 dim pose representation
        opt.audio_dir = pjoin(opt.data_root, 'extracted_audios')
        opt.joints_num = 24


    elif opt.dataset_name == 'aioz':
        opt.data_root = './dataset/AIOZ_Gdance_dataset'
        opt.motion_dir = pjoin(opt.data_root, 'annotations/motions') # using smpl 72 dim pose representation
        opt.audio_dir = pjoin(opt.data_root, 'extracted_audios')
        opt.joints_num = 24


    elif opt.dataset_name == 'aamixed':
        opt.data_root = './dataset/AIOZ_Gdance_dataset'
        opt.motion_dir = pjoin(opt.data_root, 'annotations/motions') # using smpl 72 dim pose representation
        opt.audio_dir = pjoin(opt.data_root, 'extracted_audios')
        opt.joints_num = 24
        

    else:
        raise KeyError('Dataset not recognized')

    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt