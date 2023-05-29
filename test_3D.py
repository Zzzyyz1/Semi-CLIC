import argparse
import os
import shutil
from glob import glob

import torch

from networks.unet_3D import unet_3D
from networks.unet_3D_dv_semi import unet_3D_dv_semi
from test_3D_Lung_util import test_all_case,test_all_case_catseg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/LungVessel_4', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Lungvessel/Lungvessel_ICLC_rca_Trainingv2s22_4', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_3D', help='model_name')


def Inference(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 3
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(64, 64, 64), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric

def Inference_v3(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 3
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net1 = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    net2 = unet_3D(n_classes=num_classes, in_channels=4).cuda()
    save_mode_path1 = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    save_mode_path2= os.path.join(
        snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
    net1.load_state_dict(torch.load(save_mode_path1))
    net2.load_state_dict(torch.load(save_mode_path2))
    print("init weight from {}".format(save_mode_path1))
    print("init weight from {}".format(save_mode_path2))
    net1.eval()
    net2.eval()
    avg_metric = test_all_case_catseg(net1,net2, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(64, 64, 64), stride_xy=64, stride_z=64, test_save_path=test_save_path)
    return avg_metric

def Inference_dv(FLAGS):
    snapshot_path = "../model/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "../model/{}/Prediction".format(FLAGS.exp)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D(n_classes=num_classes, in_channels=1).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(32, 128, 128), stride_xy=64, stride_z=16, test_save_path=test_save_path)
    return avg_metric



if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference_v3(FLAGS)
    print(metric)
    print((metric[0][0]+metric[1][0])/2)
