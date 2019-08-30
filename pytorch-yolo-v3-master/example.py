from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random
import pickle as pkl
import itertools


class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers = num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5, 5) for x in range(num_layers)])
        self.output = nn.Linear(5, 2)

    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()
    num_classes
    return img_


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()

    scales = args.scales

    #        scales = [int(x) for x in scales.split(',')]
    #
    #
    #
    #        args.reso = int(args.reso)
    #
    #        num_boxes = [args.reso//32, args.reso//16, args.reso//8]
    #        scale_indices = [3*(x**2) for x in num_boxes]
    #        scale_indices = list(itertools.accumulate(scale_indices, lambda x,y : x+y))
    #
    #
    #        li = []
    #        i = 0
    #        for scale in scale_indices:
    #            li.extend(list(range(i, scale)))
    #            i = scale
    #
    #        scale_indices = li

    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    classes = load_classes('data/coco.names')

    # Set up the neural network
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    read_dir = time.time()
    # Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if
                  os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[
                      1] == '.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print("No file or directory with the name {}".format(images))
        exit()

    if not os.path.exists(args.det):
        os.makedirs(args.det)

    load_batch = time.time()

    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    leftover = 0

    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    i = 0

    write = False
    model(get_test_input(inp_dim, CUDA), CUDA)

    start_det_loop = time.time()

    objs = {}

    for batch in im_batches:
        # load the image
        start = time.time()
        if CUDA:
            batch = batch.cuda()

        # Apply offsets to the result predictions
        # Tranform the predictions as described in the YOLO paper
        # flatten the prediction vector
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        #        prediction = prediction[:,scale_indices]

        # get the boxes with object confidence > threshold
        # Convert the cordinates to absolute coordinates
        # perform NMS on these boxes, and save the results
        # I could have done NMS and saving seperately to have a better abstraction
        # But both these operations require looping, hence
        # clubbing these ops in one loop instead of two.
        # loops are slower than vectorised operations.

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(prediction) == int:
            i += 1
            continue

        end = time.time()

        #        print(end - start)

        prediction[:, 0] += i * batch_size

        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output, prediction))

        for im_num, image in enumerate(imlist[i * batch_size: min((i + 1) * batch_size, len(imlist))]):
            im_id = i * batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start) / batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        if CUDA:
            torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No detections were made")
        exit()

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(inp_dim / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    output_recast = time.time()

    class_load = time.time()

    colors = pkl.load(open("pallete", "rb"))

    draw = time.time()


    def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        cls = int(x[-1])
        label = "{0}".format(classes[cls])
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
        return img


    list(map(lambda x: write(x, im_batches, orig_ims), output))

    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det, x.split("/")[-1]))

    list(map(cv2.imwrite, det_names, orig_ims))

    end = time.time()

    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) + " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch) / len(imlist)))
    print("----------------------------------------------------------")

    torch.cuda.empty_cache()







    ################################################################

    # !/usr/bin/env python[3.6]

    from ctypes import *
    import math
    import random
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image
    import numpy as np
    import pymysql


    def sample(probs):
        s = sum(probs)
        probs = [a / s for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs) - 1


    def c_array(ctype, values):
        arr = (ctype * len(values))()
        arr[:] = values
        return arr


    class BOX(Structure):
        _fields_ = [("x", c_float),
                    ("y", c_float),
                    ("w", c_float),
                    ("h", c_float)]


    class DETECTION(Structure):
        _fields_ = [("bbox", BOX),
                    ("classes", c_int),
                    ("prob", POINTER(c_float)),
                    ("mask", POINTER(c_float)),
                    ("objectness", c_float),
                    ("sort_class", c_int)]


    class IMAGE(Structure):
        _fields_ = [("w", c_int),
                    ("h", c_int),
                    ("c", c_int),
                    ("data", POINTER(c_float))]


    class METADATA(Structure):
        _fields_ = [("classes", c_int),
                    ("names", POINTER(c_char_p))]


    # lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
    lib = CDLL("/home/team/darknet/libdarknet.so", RTLD_GLOBAL)
    lib.network_width.argtypes = [c_void_p]
    lib.network_width.restype = c_int
    lib.network_height.argtypes = [c_void_p]
    lib.network_height.restype = c_int

    predict = lib.network_predict
    predict.argtypes = [c_void_p, POINTER(c_float)]
    predict.restype = POINTER(c_float)

    set_gpu = lib.cuda_set_device
    set_gpu.argtypes = [c_int]

    make_image = lib.make_image
    make_image.argtypes = [c_int, c_int, c_int]
    make_image.restype = IMAGE

    get_network_boxes = lib.get_network_boxes
    get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
    get_network_boxes.restype = POINTER(DETECTION)

    make_network_boxes = lib.make_network_boxes
    make_network_boxes.argtypes = [c_void_p]
    make_network_boxes.restype = POINTER(DETECTION)

    free_detections = lib.free_detections
    free_detections.argtypes = [POINTER(DETECTION), c_int]

    free_ptrs = lib.free_ptrs
    free_ptrs.argtypes = [POINTER(c_void_p), c_int]

    network_predict = lib.network_predict
    network_predict.argtypes = [c_void_p, POINTER(c_float)]

    reset_rnn = lib.reset_rnn
    reset_rnn.argtypes = [c_void_p]

    load_net = lib.load_network
    load_net.argtypes = [c_char_p, c_char_p, c_int]
    load_net.restype = c_void_p

    do_nms_obj = lib.do_nms_obj
    do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    do_nms_sort = lib.do_nms_sort
    do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

    free_image = lib.free_image
    free_image.argtypes = [IMAGE]

    letterbox_image = lib.letterbox_image
    letterbox_image.argtypes = [IMAGE, c_int, c_int]
    letterbox_image.restype = IMAGE

    load_meta = lib.get_metadata
    lib.get_metadata.argtypes = [c_char_p]
    lib.get_metadata.restype = METADATA

    load_image = lib.load_image_color
    load_image.argtypes = [c_char_p, c_int, c_int]
    load_image.restype = IMAGE

    rgbgr_image = lib.rgbgr_image
    rgbgr_image.argtypes = [IMAGE]

    predict_image = lib.network_predict_image
    predict_image.argtypes = [c_void_p, IMAGE]
    predict_image.restype = POINTER(c_float)


    def classify(net, meta, im):
        out = predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res


    def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): do_nms_obj(dets, num, meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    print(str(dets[j].prob[i]))
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        free_image(im)
        free_detections(dets, num)
        return res


    if __name__ == "__main__":

        db = pymysql.connect(host="13.124.223.128", user="chicken", password="Chicken18@",
                             db='db', charset='utf8')
        cur = db.cursor()
        # db = pymysql.connect(host="54.180.168.210", user="root", password="rlawhddud1!",
        #                      db='sns', charset='utf8')
        # cur = db.cursor()
        person = []
        chair = []
        # print("시작")
        # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
        # im = load_image("data/wolf.jpg", 0, 0)
        # meta = load_meta("cfg/imagenet1k.data")
        # r = classify(net, meta, im)
        # print r[:10]
        net = load_net(b"/home/team/darknet/cfg/yolov3.cfg", b"/home/team/darknet/yolov3.weights", 0)
        meta = load_meta(b"/home/team/darknet/cfg/coco.data")
        r = detect(net, meta, b"/home/team/darknet/python/2.jpg")
        print(str(r))
        # print("ㅂㅂㅂ")
        im2 = np.array(Image.open(b"/home/team/darknet/python/2"
                                  b".jpg"), dtype=np.uint8)
        fig, ax = plt.subplots(1)
        font_lo = "/usr/share/fonts/truetype/NanumGothic.ttf"
        font_name = fm.FontProperties(fname=font_lo).get_name()
        import matplotlib.font_manager as fm
        import matplotlib

        fontprop = fm.FontProperties(fname=font_lo, size=10)
        matplotlib.rc('font', family=font_name)

        # fig.set_size_inches(imgw,imgh)
        ax.imshow(im2)
        # rc('font',family='sans-serif')
        for k in range(len(r)):
            # print r[k][2][0]
            # print r[k][2][3]
            width = r[k][2][2]
            height = r[k][2][3]
            center_x = r[k][2][0]
            center_y = r[k][2][1]
            bottomLeft_x = center_x - (width / 2)
            bottomLeft_y = center_y - (height / 2)
            # rect = patches.Rectangle((r[k][2][0],r[k][2][3]),r[k][2][2],r[k][2][3],linewidth=1,edgecolor='r',facecolor='none')
            name = str(r[k][0]).replace("b'", "").replace("'", "")
            if (name == 'chair') or (name == 'person'):
                if (name == 'chair'):
                    chair.append(str(name) + "/" + str(width) + "/" + str(height) + "/" + str(center_x) + "/" + str(
                        center_y) + "/" + str(bottomLeft_x) + "/" + str(bottomLeft_y))
                else:
                    person.append(str(name) + "/" + str(width) + "/" + str(height) + "/" + str(center_x) + "/" + str(
                        center_y) + "/" + str(bottomLeft_x) + "/" + str(bottomLeft_y))

        chair = (sorted(chair, key=lambda x: float(x.split("/")[3])))
        person = (sorted(person, key=lambda x: float(x.split("/")[3])))

        for i, v in enumerate(chair):
            color = "r"
            bottomLeft_x = float(chair[i].split("/")[5])
            bottomLeft_y = float(chair[i].split("/")[6])
            name = str(chair[i].split("/")[0])
            width = float(chair[i].split("/")[1])
            height = float(chair[i].split("/")[2])
            center_x = float(chair[i].split("/")[3])
            # print(str(center_x))
            print(str(bottomLeft_x) + " " + str(bottomLeft_y + height))
            center_y = float(chair[i].split("/")[4])
            # 의자 표시
            rect = patches.Rectangle((bottomLeft_x, bottomLeft_y), width, height, linewidth=3, edgecolor=str(color),
                                     facecolor='none')
            # 중앙점 표시
            rect2 = patches.Circle((center_x - 1, center_y - 1), 5, facecolor=str(color), edgecolor=str(color))

            plt.text(center_x + 7, center_y + 10, "(" + str(int(center_x)) + "," + str(int(center_y)) + ")",
                     color=str(color), size="7")
            plt.text(center_x - (width / 2) + 15, center_y - (height / 2), str("의자 ") + str(i), color="white",
                     weight=1000,
                     bbox=dict(facecolor=str(color), alpha=1, edgecolor=str(color)), fontproperties=fontprop)

            ax.add_patch(rect)
            ax.add_patch(rect2)

            # ///
            # 의자 왼쪽 하단 점 표시
            rect3 = patches.Circle((bottomLeft_x + (width / 4), bottomLeft_y - 50 + height), 5, facecolor=str(color),
                                   edgecolor=str(color))
            # 의자 오른쪽 하단 점 표시
            rect4 = patches.Circle((bottomLeft_x + (width / 4) * 3, bottomLeft_y - 50 + height), 5,
                                   facecolor=str(color),
                                   edgecolor=str(color))
            ax.add_patch(rect3)
            ax.add_patch(rect4)

            # sql="insert into jj(c0,c1,c2,c3) values (%s,%s,%s,%s)"

            sql = "update posit set position1X=%s, position1Y =%s, position2X =%s, position2Y =%s where (position1X>=%s and position1X<=%s) and (position2X >= %s and position2X <= %s) " \
                  "and (position1Y>=%s and position1Y<=%s) and " \
                  "(position2Y>=%s and position2Y<=%s) "
            cur.execute(sql, ((int(bottomLeft_x + (width / 4))), (int(bottomLeft_y + height - 50)),
                              (int(bottomLeft_x + (width / 4) * 3)), (int(bottomLeft_y + height - 50)),

                              (int(bottomLeft_x + (width / 4) - 100)), (int(bottomLeft_x + (width / 4) + 100)),
                              (int(bottomLeft_x + (width / 4) * 3 - 100)), (int(bottomLeft_x + (width / 4) * 3 + 100)),

                              (int(bottomLeft_y + height - 200)), (int(bottomLeft_y + height + 100)),
                              (int(bottomLeft_y + height - 200)), (int(bottomLeft_y + height + 100))))
            #
            # sql = "insert posit values(%s,%s,%s,%s,%s,%s)"
            # cur.execute(sql, (str(i),str(int(bottomLeft_x + (width / 4))), str(int(bottomLeft_y + height - 50)),
            #                   str(int(bottomLeft_x + (width / 4) * 3)), str(int(bottomLeft_y + height - 50)), 0))
            #
            # print("ㄴㄴㄴㄴㄴㄴㄴㄴㄴ")
            # print(str(sql)
            # print(str(int(bottomLeft_x + (width / 4)))+" "+ str(int(bottomLeft_y + height - 50))+" "+str(int(bottomLeft_x + (width / 4) * 3))+" "+ str(int(bottomLeft_y + height - 50))+" "+)

        sql = "update posit set check_=0"
        cur.execute(sql)
        for i, v in enumerate(person):
            color = "b"
            bottomLeft_x = float(person[i].split("/")[5])
            bottomLeft_y = float(person[i].split("/")[6])
            name = str(person[i].split("/")[0])
            width = float(person[i].split("/")[1])
            height = float(person[i].split("/")[2])
            center_x = float(person[i].split("/")[3])
            center_y = float(person[i].split("/")[4])
            rect = patches.Rectangle((bottomLeft_x, bottomLeft_y), width, height, linewidth=3, edgecolor=str(color),
                                     facecolor='none')
            rect2 = patches.Circle((center_x - 1, center_y - 1), 5, facecolor=str(color), edgecolor=str(color))

            plt.text(center_x + 7, center_y + 10, "(" + str(int(center_x)) + "," + str(int(center_y)) + ")",
                     color=str(color), size="7")
            plt.text(center_x - (width / 2) + 15, center_y - (height / 2), str("사람 ") + str(i), color="white",
                     weight=1000,
                     bbox=dict(facecolor=str(color), alpha=1, edgecolor=str(color)), fontproperties=fontprop)

            ax.add_patch(rect)
            ax.add_patch(rect2)
            sql = "select * from posit"
            cur.execute(sql)
            result = cur.fetchall()
            for i, re in enumerate(result):
                bottomLeft_yy = bottomLeft_y + height
                if ((int(re[1]) <= int(center_x) and int(center_x) <= int(re[3])) and int(bottomLeft_y + height) >= int(
                        re[4])):
                    plt.text(center_x + 7, center_y + 50, "(있음)",
                             color=str("w"), size="7")
                    sql = "update posit set check_=1 where seatnumber=" + re[0]
                    cur.execute(sql)
                    break
                    # return

                if (i == 3):
                    plt.text(center_x + 7, center_y + 50, "(없음)",
                             color=str("w"), size="7")
                    # 전 이미지 하단 좌
                    rect3 = patches.Circle((bottomLeft_x + width / 4, bottomLeft_y + height), 5, facecolor=str(color),
                                           edgecolor=str("r"))
                    rect4 = patches.Circle((bottomLeft_x + (width / 4) * 3, bottomLeft_y + height), 5,
                                           facecolor=str(color),
                                           edgecolor=str("r"))
                    ax.add_patch(rect3)
                    ax.add_patch(rect4)
                    sql = "update posit set check_=0 where seatnumber=" + re[0]
                    cur.execute(sql)

            # # 사람의 하단 좌표
            rect3 = patches.Circle((bottomLeft_x + (width / 4), bottomLeft_y + height), 5, facecolor=str(color),
                                   edgecolor=str(color))
            rect4 = patches.Circle((bottomLeft_x + (width / 4) * 3, bottomLeft_y + height), 5, facecolor=str(color),
                                   edgecolor=str(color))
            ax.add_patch(rect3)
            ax.add_patch(rect4)

        # sql="select * from posit"
        # cur.execute(sql)
        # r=cur.fetchall()
        # for i,rr in enumerate(r):
        #     rect3 = patches.Circle((float(rr[1]),float(rr[2])), 5, facecolor=str(color),
        #                            edgecolor=str("y"))
        #     rect4 = patches.Circle((float(rr[3]), float(rr[4])), 5, facecolor=str(color),
        #                            edgecolor=str("y"))
        #     ax.add_patch(rect3)
        #     ax.add_patch(rect4)

        db.commit()
        ax.xaxis.set_ticks_position('top')
        # plt.gca().axes.get_xaxis().set_visible(False)
        plt.savefig('/home/team/darknet/python/2_result.jpg', dpi=400)