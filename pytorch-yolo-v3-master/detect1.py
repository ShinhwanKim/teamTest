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
import pymysql
import itertools
import urllib.request

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
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
    
    file = urllib.request.urlretrieve('http://13.124.223.128/image/result_1.jpg', 'result_1.jpg')

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "result_1.jpg", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--scales", dest = "scales", help = "Scales to use for detection",
                        default = "1,2,3", type = str)
    
    return parser.parse_args()

if __name__ ==  '__main__':

    db = pymysql.connect(host='13.124.223.128', user='chicken',
                         passwd='Chicken18@', db='db', charset='utf8')
    cur = db.cursor()
    while True:

        while True:
            sql='select image1 from wait'
            cur.execute(sql)
            re=cur.fetchall()
            for v in re:
                if(re[0]==1):
                    break;
                else :
                    db.commit()

        args = arg_parse()
        scales = args.scales
        images = args.images
        batch_size = int(args.bs)
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        model.net_info["height"] = args.reso



        person=[]
        chair=[]




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


        start = 0

        CUDA = torch.cuda.is_available()

        num_classes = 80
        classes = load_classes('data/coco.names')

        #Set up the neural network
        print("Loading network.....")

        print("Network successfully loaded")

        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32

        #If there's a GPU availible, put the model on GPU
        if CUDA:
            model.cuda()


        #Set the model in evaluation mode
        model.eval()

        read_dir = time.time()
        #Detection phase
        try:
            imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
        except NotADirectoryError:
            imlist = []
            imlist.append(osp.join(osp.realpath('.'), images))
        except FileNotFoundError:
            print ("No file or directory with the name {}".format(images))
            exit()

        if not os.path.exists(args.det):
            os.makedirs(args.det)

        load_batch = time.time()

        batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
        im_batches = [x[0] for x in batches]
        orig_ims = [x[1] for x in batches]
        im_dim_list = [x[2] for x in batches]
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)



        if CUDA:
            im_dim_list = im_dim_list.cuda()

        leftover = 0

        if (len(im_dim_list) % batch_size):
            leftover = 1


        if batch_size != 1:
            num_batches = len(imlist) // batch_size + leftover
            im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                                len(im_batches))]))  for i in range(num_batches)]


        i = 0


        write = False
        model(get_test_input(inp_dim, CUDA), CUDA)

        start_det_loop = time.time()

        objs = {}



        for batch in im_batches:
            #load the image
            start = time.time()
            if CUDA:
                batch = batch.cuda()


            #Apply offsets to the result predictions
            #Tranform the predictions as described in the YOLO paper
            #flatten the prediction vector
            # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes)
            # Put every proposed box as a row.
            with torch.no_grad():
                prediction = model(Variable(batch), CUDA)

    #        prediction = prediction[:,scale_indices]


            #get the boxes with object confidence > threshold
            #Convert the cordinates to absolute coordinates
            #perform NMS on these boxes, and save the results
            #I could have done NMS and saving seperately to have a better abstraction
            #But both these operations require looping, hence
            #clubbing these ops in one loop instead of two.
            #loops are slower than vectorised operations.

            prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)


            if type(prediction) == int:
                i += 1
                continue

            end = time.time()


    #        print(end - start)



            prediction[:,0] += i*batch_size




            if not write:
                output = prediction
                write = 1
            else:
                # if output.size()[1] == prediction.size()[1]:
                output = torch.cat((output,prediction))




            for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
                im_id = i*batch_size + im_num
                objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
                print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
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

        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)


        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])


        output_recast = time.time()


        class_load = time.time()

        colors = pkl.load(open("pallete", "rb"))


        draw = time.time()


        def write(x, batches, results):

            c1 = tuple(x[1:3].int())
            print(str(x[1:3]))
            c2 = tuple(x[3:5].int())
            # img = results[int(x[0])]
            img=int(x[0])
            cls = int(x[-1])
            if str(format(classes[cls]))=='chair' or str(format(classes[cls]))=='person':
                label = "{0}".format(classes[cls])
                name=""+str(format(classes[cls]))


                print(str(format(classes[cls])))
                for r in x:
                    print(str(r).replace("tensor(","").replace(")",""))
                print("\n")



                cv2.rectangle(img, c1, c2,[255,255,0], 4)

                width = c2[0] - c1[0]
                height = c1[1] - c2[1]
                center_x = c1[0] + (width/2)
                center_y = c1[1] - (height/2)
                check_x1 = c1[0] + (width/4)
                check_x2 = c2[0] - (width/4)
                check_y = c2[1] + height/5
                cv2.line(img, (center_x,center_y), (center_x,center_y), [255,0,0], 15)
                if(name=="person"):
                    person.append(str(name)+"/"+str(c1[0]).replace("tensor(","").split(",")[0]+"/"+str(c1[1]).replace("tensor(","").split(",")[0]+"/"+
                                  str(c2[0]).replace("tensor(", "").split(",")[0] + "/"+str(c2[1]).replace("tensor(","").split(",")[0]+"/"+
                                  str(center_x).replace("tensor(","").split(",")[0]+"/"+ str(center_y).replace("tensor(","").split(",")[0]+"/"+
                                  str(check_x1).replace("tensor(","").split(",")[0]+"/"+str(check_x2).replace("tensor(","").split(",")[0]+"/"+
                                  str(check_y).replace("tensor(","").split(",")[0]+"/"+str(width).replace("tensor(","").split(",")[0]+"/"+
                                  str(height).replace("tensor(","").split(",")[0]+"/"+ str(img))
                else:
                    #chair.append(str(name)+"/"+str(c1[0])+"/"+str(c1[1])+"/"+str(center_x)+"/"+str(center_y)+"/"+str(width)+"/"+str(height))
                    chair.append(str(name)+"/"+str(c1[0]).replace("tensor(","").split(",")[0]+"/"+str(c1[1]).replace("tensor(","").split(",")[0]+"/"+
                                  str(c2[0]).replace("tensor(", "").split(",")[0] + "/"+str(c2[1]).replace("tensor(","").split(",")[0]+"/"+
                                  str(center_x).replace("tensor(","").split(",")[0]+"/"+ str(center_y).replace("tensor(","").split(",")[0]+"/"+
                                  str(check_x1).replace("tensor(","").split(",")[0]+"/"+str(check_x2).replace("tensor(","").split(",")[0]+"/"+
                                  str(check_y).replace("tensor(","").split(",")[0]+"/"+str(width).replace("tensor(","").split(",")[0]+"/"+
                                  str(height).replace("tensor(","").split(",")[0]+"/"+ str(img))

                #print(chair) -> ['chair/647/686/1088/943/882/-514', 'chair/1570/942/1743/1068/347/-

                #c1의 x좌표로 정렬(낮은순서대로)


                # for i in chair:
                #     print(str(i)+"\n")
                # sql="select count(*) as sehoon from posit where seatnumber <=3"
                # cur.execute(sql)
                # num=cur.fetchall()
                # if(int(num[0])!=4):
                #     sql="delete from posit where seatnumber<=3"
                #     cur.execute(sql)
                # else:
                #     sql = "update posit set positon1X=%s, position1Y=%s, position2X=%s, positio2Y=%s where (position1X>=%s and position1X <%s) and (position2X>=%s and position2X<%s) " \
                #     "and (position1Y>=%s and position2Y<=%s)"
                #     cur.execute(sql,())

                # key = cv2.waitKey(0)

        list(map(lambda x: write(x, im_batches, orig_ims), output))
        chair.sort(key=lambda x: float(x.split("/")[1]))
        person.sort(key=lambda x: float(x.split("/")[1]))
        #print(chair)
        for i,v in enumerate(chair):
            label = str(chair[i].split("/")[0] + str(i))
            Left_x1 = int(chair[i].split("/")[1])
            Left_y1 = int(chair[i].split("/")[2])
            Right_x2 = int(chair[i].split("/")[3])
            Right_y2 = int(chair[i].split("/")[4])
            center_x = int(chair[i].split("/")[5])
            center_y = int(chair[i].split("/")[6])
            check_x1 = int(chair[i].split("/")[7])
            check_x2 = int(chair[i].split("/")[8])
            check_y = int(chair[i].split("/")[9])
            width = int(chair[i].split("/")[10])
            height = int(chair[i].split("/")[11])
            img = orig_ims[int(chair[i].split("/")[12])]

            cv2.line(img, (Left_x1,Left_y1), (Left_x1, Left_y1), [0,255,0], 20)
            cv2.line(img, (Right_x2, Right_y2), (Right_x2, Right_y2), [0, 255, 0], 20)

            # print(img)
            cv2.rectangle(img, (Left_x1,Left_y1), (Right_x2,Right_y2), [255,0,0], 4)
            # cv2.rectangle(img, (Left_x1,Left_y1), (Right_x2,Right_y2), [255,0,0], -1)
            cv2.line(img, (check_x1, check_y), (check_x1, check_y), [0, 0, 255], 10)
            cv2.line(img, (check_x2, check_y), (check_x2, check_y), [0, 0, 255], 10)
            cv2.line(img, (center_x, center_y), (center_x, center_y), [0, 255, 0], 15)
            #
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = Left_x1 + t_size[0] + 3, Left_y1 + t_size[1] + 4
            cv2.rectangle(img, (Left_x1, Left_y1), (Left_x1 + (len(label) * 30), Left_y1 + 50), [255, 0, 255], -1)
            cv2.putText(img, label, (Left_x1, Left_y1 + t_size[1] + 30), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], 3)
            # path = ""
            # cv2.imwrite(os.path.join(path, "1.jpg"), img)
            # sql="insert into jj(c0,c1,c2,c3) values (%s,%s,%s,%s)"

            sql = "update posit set position1X=%s, position1Y =%s, position2X =%s, position2Y =%s where (position1X>=%s and position1X<=%s) and (position2X >= %s and position2X <= %s) " \
                  "and (position1Y>=%s and position1Y<=%s) and " \
                  "(position2Y>=%s and position2Y<=%s) and (seatnumber<=3)"
            cur.execute(sql, ((int(check_x1), (int(Right_y2)),
                              (int(check_x2)), (int(Right_y2)),

                              # 조건
                              (int(check_x1 - 100)), (int(check_x1 + 100)),
                              (int(check_x2 - 100)), (int(check_x2 + 100)),

                              (int(Right_y2 - 200)), (int(Right_y2 + 200)),
                              (int(Right_y2 - 200)), (int(Right_y2 + 200)))))
            print("시작")

            print(str(check_x1)+" "+str(Right_y2))
            print(str(check_x2)+" "+str(Right_y2))
            print("")
            print("조건")
            print(str(int(check_x1)-100)+" "+str(int(check_x1 + 100)))
            print(str(int(check_x2 - 100))+" "+str(int(check_x2 + 100)))
            print("")
            print(str(int(Right_y2-100))+" "+str(int(Right_y2+100)))
            print(str(int(Right_y2-100))+" "+str(int(Right_y2+100)))
            print("끝")
            #
            # sql = "insert posit values(%s,%s,%s,%s,%s,%s)"
            # cur.execute(sql, (i,str(int(check_x1)), str(int(Right_y2)),
            #                   str(int(check_x2)), str(int(Right_y2)), 0))


        sql="update posit set check_=0 where seatnumber <=3"
        cur.execute(sql)
        for i,v in enumerate(person):
            label = str(person[i].split("/")[0] + str(i))
            Left_x1 = int(person[i].split("/")[1])
            Left_y1 = int(person[i].split("/")[2])
            Right_x2 = int(person[i].split("/")[3])
            Right_y2 = int(person[i].split("/")[4])
            center_x = int(person[i].split("/")[5])
            center_y = int(person[i].split("/")[6])
            check_x1 = int(person[i].split("/")[7])
            check_x2 = int(person[i].split("/")[8])
            check_y = int(person[i].split("/")[9])
            width = int(person[i].split("/")[10])
            height = int(person[i].split("/")[11])
            img = orig_ims[int(person[i].split("/")[12])]


            # print(img)
            cv2.rectangle(img, (Left_x1,Left_y1), (Right_x2,Right_y2), [0,255,0], 4)
            # cv2.rectangle(img, (Left_x1,Left_y1), (Right_x2,Right_y2), [255,0,0], -1)
            cv2.line(img, (check_x1, check_y), (check_x1, check_y), [0, 0, 255], 10)
            cv2.line(img, (check_x2, check_y), (check_x2, check_y), [0, 0, 255], 10)
            cv2.line(img, (center_x, center_y), (center_x, center_y), [0, 255, 0], 15)
            #
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = Left_x1 + t_size[0] + 3, Left_y1 + t_size[1] + 4
            cv2.rectangle(img, (Left_x1, Left_y1), (Left_x1 + (len(label) * 30), Left_y1 + 50), [255, 0, 255], -1)
            cv2.putText(img, label, (Left_x1, Left_y1 + t_size[1] + 30), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], 3)
            path = ""
            # cv2.imwrite(os.path.join(path, "1.jpg"), img)
            sql="select * from posit where seatnumber <=3"
            cur.execute(sql)
            result=cur.fetchall()
            for i, re in enumerate(result):
                print("쿼리문")
                if ((int(re[1]) <= int(center_x) and int(center_x) <= int(re[3])) and int(Right_y2) >= int(re[4])):
                    cv2.putText(img, "있음", (center_x+10, center_y+50), cv2.FONT_HERSHEY_PLAIN, 3, [225, 255, 255], 3)
                    # plt.text(center_x + 7, center_y + 50, "(있음)",color=str("w"), size="7")
                    sql = "update posit set check_=1 where seatnumber=%s"
                    print("갱신")
                    cur.execute(sql,(int(re[0])))
                    break

        det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))

        list(map(cv2.imwrite, det_names, orig_ims))
        cv2.imwrite(os.path.join("", "1.jpg"), img)

        end = time.time()
        sql="update wait set image1=0"
        cur.execute(sql)
        db.commit()

        print()
        print("SUMMARY")
        print("----------------------------------------------------------")
        print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
        print()
        print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
        print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
        print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
        print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
        print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
        print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
        print("----------------------------------------------------------")


        torch.cuda.empty_cache()
    
    
        
        
    
    
