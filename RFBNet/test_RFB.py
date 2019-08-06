from __future__ import print_function
import sys
import os
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import AnnotationTransform, COCODetection, VOCDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300
import cv2
import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFBNet300_VOC_80_7.pth',
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


def test_net(im, net, detector, cuda, transform, max_per_image=3000, thresh=0.005):

    #if not os.path.exists(save_folder):
     #   os.mkdir(save_folder)
    num_images = 1
    num_classes = (21, 21)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    #cap = cv2.VideoCapture(0)

    i=0

    _t = {'im_detect': Timer(), 'misc': Timer()}
    #det_file = os.path.join(save_folder, 'detections.pkl')

   # while(True):
    #ret,img = cap.read()
        #img = testset.pull_image(i)
    img = im
    
    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()

    _t['im_detect'].tic()
    out = net(x)  
    boxes, scores = detector.forward(out,priors)
    detect_time = _t['im_detect'].toc()
    boxes = boxes[0]
    scores=scores[0]

    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()


    for j in range(1, num_classes):           
        inds = np.where(scores[:, j] > thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
            #none1 = np.array([[20,20,30,30,0.98]])
            #return none1
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(c_dets, 0.45, force_cpu=args.cpu)
        c_dets = c_dets[keep, :]
        all_boxes[j][i] = c_dets        
    return all_boxes[15][i]

    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]
    
'''
    if(ret):   
        img = cv2.resize(img, (740, 500))
        cv2.imshow('name',img)
    else: 
        break 
    if cv2.waitKey(1)==27: 
        break 
    i = 0

'''



if __name__ == '__main__':
    # load net
    img_dim = (300,512)[args.size=='512']
    num_classes = (21, 21)[args.dataset == 'COCO']
    net = build_net('test', img_dim, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

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
    print(net)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    top_k = 200
    detector = Detect(num_classes,0,cfg)
    #save_folder = os.path.join(args.save_folder,args.dataset)
    rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
    cap = cv2.VideoCapture(0)
    
    while(True):
        test_net(img, net, detector, args.cuda, 
                 BaseTransform(net.size, rgb_means, (2, 0, 1)),
                 top_k, thresh=0.4)
        



