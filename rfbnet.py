from __future__ import print_function
import os
import cv2
import numpy as np
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import sys
sys.path.append('/home/zzw/桌面/centerNet-deep-sort-master/RFBNet')
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
from models.RFB_Net_vgg import build_net



from test_RFB import test_net



from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

import time


parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='RFBNet/weights/RFBNet300_VOC_80_7.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


img_dim = (300,512)[args.size=='512']
num_classes = (21, 21)[args.dataset == 'COCO']
net = build_net('test', img_dim, num_classes)    # initialize detector
state_dict = torch.load(args.trained_model)

from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()
print('Finished loading model!')

if args.cuda:
    net = net.cuda()
    cudnn.benchmark = True
else:
    net = net.cpu()

top_k = 200
detector = Detect(num_classes,0,cfg)
#save_folder = os.path.join(args.save_folder,args.dataset)
rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']





def bbox_to_xywh_cls_conf(bbox):
    person_id = 1
    confidence = 0.4
    # only person
    #bbox = bbox[person_id]

    if any(bbox[:, 4] > confidence):
    #if len(bbox[:, 4]) >= 1:

        bbox = bbox[bbox[:, 4] > confidence, :]
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]  
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]  
        return bbox[:, :4], bbox[:, 4]

    else:
        return None, None


class Detector(object):

    def __init__(self):
        self.vdo = cv2.VideoCapture()
        self.deepsort = DeepSort("deep/checkpoint/ckpt.t7")
        self.write_video = True

    def open(self, video_path):

        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter("demo1.avi", fourcc, 20, (self.im_width, self.im_height))

        return self.vdo.isOpened()



    def detect(self):
        xmin, ymin, xmax, ymax = self.area
        frame_no = 0
        avg_fps = 0.0
        
        while self.vdo.grab():
            frame_no +=1
            _, ori_im = self.vdo.retrieve()
            im = ori_im[ymin:ymax, xmin:xmax]



            results = test_net(im, net, detector, args.cuda, 
                     BaseTransform(net.size, rgb_means, (2, 0, 1)),
                     top_k, thresh=0.4)                      
            # RFBNet使用教程
            bbox_xywh, cls_conf = bbox_to_xywh_cls_conf(results)  

            if bbox_xywh is not None:
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))
                    
            cv2.imshow("test", ori_im)
            cv2.waitKey(1)

            if self.write_video:
               self.output.write(ori_im)


if __name__ == "__main__":
    import sys


    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("test", 700, 580)

    #opt = opts().init()
    det = Detector()

    # det.open("D:\CODE\matlab sample code/season 1 episode 4 part 5-6.mp4")
    det.open("123.mp4")
    det.detect()

