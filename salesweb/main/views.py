# -*- coding: utf-8 -*-

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.template import RequestContext
from django.views.generic import View, TemplateView, DetailView
from decimal import Decimal
from .models import TrainingModel, ResultModel
from dateutil.relativedelta import relativedelta
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from datetime import datetime

import pandas as pd
import os
import json
import sys
import subprocess
import psutil
import ast

## model selection
from django.db.models import Q
from django.core import serializers

from os import listdir
from os.path import isfile, join
from shutil import copyfile, copy2, rmtree
from ipware import get_client_ip
import cx_Oracle

from distutils.version import StrictVersion
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image, ImageGrab
import cv2

# from camera import VideoCamera
# import cv2

BASE_DIR = os.getcwd()

class index(TemplateView):
    template_name = "main/index.html"

@method_decorator(csrf_exempt, name='dispatch')
class getModels(TemplateView):
    def post(self, request):
        models = TrainingModel.objects.all()

        modelList = []
        for model in models:
            middle = {}
            middle['MODELNAME'] = model.modelname
            middle['PID'] = model.pid
            middle['LABEL'] = model.label
            middle['METHOD'] = model.method
            middle['CREATED_AT'] = str(model.created_at)

            if psutil.pid_exists(int(middle['PID'])):
                middle['STATUS'] = 1
            else:
                middle['STATUS'] = 0

            modelList.append(middle)

        data = json.dumps(modelList)
        return HttpResponse(data)

class manageModel(TemplateView):
    template_name = 'main/object_detection/manageModel.html'
    
    
class trainModel(TemplateView):
    template_name = 'main/object_detection/trainModel.html'    
    
    def post(self, request):
        labels = request.POST.getlist('labels')
        files = request.FILES.getlist('files')

        ## Create CSV from Image
        modelname = 'model' + datetime.now().strftime('%y%m%d%H%M%S')

        modeldirpath = os.path.join(BASE_DIR + settings.MEDIA_URL, modelname)

        try:
            if not os.path.exists(modeldirpath):
                os.makedirs(modeldirpath)
        except OSError:
            print('Error: Creating directory. ' + modeldirpath)

        col = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
        df = pd.DataFrame(columns=col)
        df.to_csv(os.path.join(modeldirpath, "train.csv"), header=True, index=False)

        datalist = []
        labelist = []
        labelidx = []

        for idx, file in enumerate(files):
            label = labels[idx + 1]

            filePath = '%s/%s' % (modelname, file.name)
            path = default_storage.save(filePath, ContentFile(file.read()))
            baseFilePath = os.path.join(BASE_DIR + settings.MEDIA_URL, path)

            width, height = Image.open(baseFilePath).size
            data = [file.name, width, height, label, 1, 1, width, height]
            datalist.append(data)

            if label not in labelist:
                labelist.append(label)

        pbtxt = ""
        for idx, label in enumerate(labelist):
            index = idx + 1
            pbtxt += "item { name:'" + label + "' id:" + str(index) + " display_name:'" + label + "' }"

            middle = {}
            middle['LABEL'] = label
            middle['IDX'] = index
            labelidx.append(middle)

        pbtxtFile = open(modeldirpath + '/train.pbtxt', 'w')
        pbtxtFile.write(pbtxt)
        pbtxtFile.close()

        labelidxjson = json.dumps(labelidx)
        labelFile = open(modeldirpath + '/labelidx.txt', 'w')
        labelFile.write(labelidxjson)
        labelFile.close()

        df = pd.DataFrame(datalist, columns=col)
        df.to_csv(os.path.join(modeldirpath, "train.csv"), header=False, index=False, mode='a')

        makeConfigFile('ssd_mobilenet_v1_coco.config', len(labelist), 30, 100, "main/object_detection/images/" + modelname + "/train.record", \
                       "main/object_detection/images/" + modelname + "/train.pbtxt", "main/object_detection/images/" + modelname + "/train.config")

        sys.stdout.flush()

        args = [modelname]
        recordProc = subprocess.Popen(["python", os.getcwd() + '/main/object_detection/generate_tfrecord.py'] + args, \
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        output, stderr = recordProc.communicate()

        outarr = output.decode('utf-8').splitlines()

        if 'True' not in outarr:
            return HttpResponse("Training Failed")

        args = ["main/object_detection/images/" + modelname + "/", "train.config"]
        trainProc = subprocess.Popen(["python", os.getcwd() + '/main/object_detection/train.py'] + args)

        newTrainingModel = TrainingModel.objects.create(
            modelname=modelname,
            pid=trainProc.pid,
            label=labelist,
            method="SSD",
        )
        newTrainingModel.save()

        return HttpResponse("Training Started Successfully")

def makeConfigFile(configFile, num_classes, batch_size, num_steps, input_path, label_map_path, out_path):

    f = open(configFile, 'r')
    rawData = f.read()
    f.close()

    rawData = json.dumps(rawData)
    rawData = json.loads(rawData)

    f = open(out_path, 'w')
    for idx, x in enumerate(rawData.split('\n')):
        if (idx == 2):
            f.write('num_classes:' + str(num_classes) + '\n')
        elif (idx == 133):
            f.write('batch_size:' + str(batch_size) + '\n')
        elif (idx == 150):
            f.write('num_steps:' + str(num_steps) + '\n')
        elif (idx == 163):
            f.write("input_path:'" + str(input_path) + "'\n")
        elif (idx == 165):
            f.write("label_map_path:'" + str(label_map_path) + "'\n")
        else:
            f.write(x + '\n')
    f.close()


class ModelCkpt(DetailView):
    def getModelChild(request):        
        method = request.GET.get("method", "")
        print("method==" + method)
        childModel = TrainingModel.objects.all().values().filter(method = method)
        childModel_list = []
        for a in childModel:
            childModel_list.append(a['modelname'])
       
        return HttpResponse(json.dumps(childModel_list), content_type='application/json')

    def getModelCkpt(request):
        parent = request.GET.get("parent", "")
        mypath = os.path.join(settings.CKPT_ROOT, parent)
        print("Trained model==" + mypath)
        
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        onlymodel = []
        latestSeq = 0
        tmpCkpt = ''
        for file in onlyfiles:
            if file[:11]=='model.ckpt-' :
                # index, meta 파일 다 있는것만 출력 --> meta 파일이 가장늦게 생성된다 이것만 찾으면 됨
                # index, meta 제외 --> meta만 찾자 
                # ckpt 숫자 가장 높은것만 - 나오는 순서가 파일명으로 sorting 되는지 모르겠으니 일단 swap 
                if file[-4:]=='meta' :
                    print(int(file[11:-5])) # sequence만 따오기
                    if latestSeq <= int(file[11:-5]) :
                        tmpCkpt = file[:-5]
        
        onlymodel.append(tmpCkpt)
        
        return HttpResponse(json.dumps(onlymodel), content_type='application/json')
        

class graphExport(TemplateView):
    template_name = 'main/object_detection/testModel.html'
    def post(self, request):
        
        child = request.POST['child']
        ckpt = request.POST['ckpt']
        
        image_path = 'c:/hsnc_od/salesweb/main/object_detection/images/'
        # image_path = settings.CKPT_ROOT
        # print(image_path)
        child_path = image_path + child
        train_graph_path =  child_path + '/train_graph'
        # train_graph export
        # train_graph 있으면 pass
        if not os.path.exists(train_graph_path):
            print('-----------export infrerence graph started-------------')
            args = ['image_tensor', child_path +'/train.config',  child_path + '/' + ckpt ,  train_graph_path ]
            print(args)

            recordProc = subprocess.Popen(["python", BASE_DIR + '/main/object_detection/export_inference_graph.py'] + args , stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            output, stderr = recordProc.communicate()
            outarr = output.decode('utf-8').splitlines()
            
            print('-----------export infrerence graph done-------------')

        # return HttpResponse("Graph Exported Successfully")

        # child = request.GET.get('child','')

        ## image list test
        ## 테스트 이미지 파일은 경로는 시간으로 분리해서 저장시켜놓고 테스트 시키자, Subprocess 에 경로만 넘겨줌
        ## 파일 경로에 있는 이미지 파일 들은 서브프로세스에서 od 한 결과 이미지로 파일 바꾸자
        print('----------object_detection test started--------------')
        files = request.FILES.getlist('files')
        nowtime = datetime.now().strftime('%y%m%d%H%M%S')
        staticImgPath = '/static/img/test/{}/{}'.format(child, nowtime)
        staticPath = 'c:/hsnc_od/salesweb'+ staticImgPath
        fpath = '{}/{}'.format(child_path, nowtime)
        filenameList = ''
        fpathList = ''
        rpathList = ''
        resultpath=[]

        try:
            if not os.path.isdir(staticPath):
                os.makedirs(os.path.join(staticPath))
        except OSError as e:
            print(e)        
        
        
        for idx, file in enumerate(files):
            filepath = '{}/{}'.format(fpath, file.name)
            rpath = '{}/{}'.format(staticPath, file.name)
            path = default_storage.save(filepath, ContentFile(file.read())) 
            tmp = Image.open(filepath)
            tmp.save(rpath)
            # fid = open(rpath, "w")
            # fid.write(ContentFile(file.read()))
            # fid.close()
            # copy2(filepath, rpath)
            filenameList += file.name + ','
            fpathList += filepath + ','
            rpathList += rpath + ','
            resultpath.append('{}/{}'.format(staticImgPath, file.name))
            print(filepath)

            

        args = [child, fpathList, rpathList, filenameList]
        print(args)
        
        testProc = subprocess.Popen(["python", BASE_DIR + '/main/object_detection/object_detection_test.py'] + args , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(os.getcwd() + '/main/object_detection/object_detection_test.py')
        output, stderr = testProc.communicate()
        outarr = output.decode('utf-8').splitlines()

        # print(outarr)
        print('saving resultmodel-------')
        for dat in outarr:
            # postgres
            outarr_dict = ast.literal_eval(dat)
            print(outarr_dict)
            ip, is_routable = get_client_ip(request)
            newResultModel = ResultModel.objects.create(
                methodname="SSD",
                modelname=child,
                datetime=nowtime,
                filename=outarr_dict.get('filename'),
                px=outarr_dict.get('xmin'),
                py=outarr_dict.get('ymin'),
                width=outarr_dict.get('xmax'),
                height=outarr_dict.get('ymax'),
                image_path=outarr_dict.get('image_path'),
                hit_yn='Y',
                ip=ip,
                label=outarr_dict.get('class'),
                score=outarr_dict.get('score')
            )
            newResultModel.save()
            # oracle S

            # oracle E

        print('-------saving resultmodel')
        # delete temp folder
        rmtree(fpath)



        print('---------object_detection test done---------------')



        return HttpResponse(json.dumps(resultpath), content_type='application/json')   

    def get_image_url():
        pass

class testCamModel(TemplateView):
    template_name = 'main/object_detection/testCamModel.html'

    def post(self, request):

        child = request.POST['child']
        ckpt = request.POST['ckpt']
        typecd = request.POST['typecd']

        image_path = 'c:/hsnc_od/salesweb/main/object_detection/images/'

        child_path = image_path + child
        train_graph_path = child_path + '/train_graph'
        # train_graph export
        # train_graph 있으면 pass
        if not os.path.exists(train_graph_path):
            print('-----------export infrerence graph started-------------')
            args = ['image_tensor', child_path + '/train.config', child_path + '/' + ckpt, train_graph_path]
            print(args)

            recordProc = subprocess.Popen(
                ["python", BASE_DIR + '/main/object_detection/export_inference_graph.py'] + args,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            output, stderr = recordProc.communicate()
            outarr = output.decode('utf-8').splitlines()

            print('-----------export infrerence graph done-------------')

        ## cam list test
        ## 테스트 이미지 파일은 경로는 시간으로 분리해서 저장시켜놓고 테스트 시키자, Subprocess 에 경로만 넘겨줌
        ## 파일 경로에 있는 이미지 파일 들은 서브프로세스에서 od 한 결과 이미지로 파일 바꾸자
        print('----------object_detection test started--------------')
        nowtime = datetime.now().strftime('%y%m%d%H%M%S')
        staticImgPath = '/static/img/test/{}/{}'.format(child, nowtime)


        args = [child]
        print(args)


        sys.path.append("..")

        from object_detection.utils import ops as utils_ops

        if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
            raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

        from utils import label_map_util

        from utils import visualization_utils as vis_util
        import cx_Oracle

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_FROZEN_GRAPH = 'c:/hsnc_od/salesweb/main/object_detection/images/{}/train_graph/frozen_inference_graph.pb'.format(child)
        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('c:\hsnc_od\salesweb\main\object_detection\images\{}'.format(child),
                                      'train.pbtxt')

        NUM_CLASSES = 2

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        def load_image_into_numpy_array(image):
            (im_width, im_height) = image.size
            if image.format == 'PNG':
                image = image.convert('RGB')
            return np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

        # # Detection
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        PATH_TO_TEST_IMAGES_DIR = 'test_images'

        IMAGE_SIZE = (12, 8)


        def run_inference_for_single_image(image, graph):
            with graph.as_default():

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict

        def save_result_model(child, label):
            pass

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:

                ret = True
                # cap = cv2.VideoCapture(0)
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1366, 768))

                if typecd == '1':
                    while (ret):
                        img = ImageGrab.grab()
                        image_np = np.array(img)

                        output_dict = run_inference_for_single_image(image_np, detection_graph)
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        # print(output_dict['detection_scores'])

                        frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        cv2.imshow('live_detection', frame)

                        if cv2.waitKey(25) & 0xFF == ord('c'):
                            cap_date = datetime.now().strftime('%y%m%d')
                            cap_time = datetime.now().strftime('%H%M%S')
                            cap_dir = "C:/hsnc_od/salesweb/main/object_detection/images/{}".format(cap_date)
                            if not os.path.exists(cap_dir):
                                os.makedirs(cap_dir)

                            cap_path = cap_dir + "/{}.png".format(cap_time)

                            cv2.imwrite(cap_path,image_np)
                            print("---- {}  saved--".format(cap_path))

                            print('saving resultmodel-------')
                            idx = 0
                            for detected in output_dict['detection_scores']:
                                if detected >= 0.80:
                                    print("====over 80%==== saving start")
                                    
                            # postgres
                                    ip, is_routable = get_client_ip(request)
                                    X = np.array(output_dict['detection_boxes'][idx]).astype(float)
                                    px = float("%0.2f"%(X[0]))
                                    py = float("%0.2f"%(X[1]))
                                    width = float("%0.2f"%(X[2]))
                                    height = float("%0.2f"%(X[3]))
                                    detected = float(np.array(detected).astype(float))

                                    newResultModel = ResultModel.objects.create(
                                        methodname="SSD",
                                        modelname=child,
                                        datetime="{}{}".format(cap_date, cap_time),
                                        filename="{}.png".format(cap_time),
                                        px=px,
                                        py=py,
                                        width=width,
                                        height=height,
                                        image_path=cap_path,
                                        hit_yn='Y',
                                        ip=ip,
                                        label=category_index[output_dict['detection_classes'][idx]].get('name'),
                                        score=detected
                                    )
                                    newResultModel.save()
                                    # oracle S
                                    print("oracle-----")


                                    # oracle E
                                    idx +=1
                            print('-------saving resultmodel')


                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            # out.release()
                            cv2.destroyAllWindows()
                            break
                else:
                    while (ret):
                        cap = cv2.VideoCapture(0)
                        ret, image_np = cap.read()
                        image_np_expanded = np.expand_dims(image_np, axis=0)

                        output_dict = run_inference_for_single_image(image_np, detection_graph)
                        # Visualization of the results of a detection.

                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)

                        cv2.imshow('cam_detection', cv2.resize(image_np, (800,600)))

                        # frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                        # cv2.imshow('live_detection', frame)


                        if cv2.waitKey(25) & 0xFF == ord('c'):
                            cap_date = datetime.now().strftime('%y%m%d')
                            cap_time = datetime.now().strftime('%H%M%S')
                            cap_dir = "C:/hsnc_od/salesweb/main/object_detection/images/{}".format(cap_date)
                            if not os.path.exists(cap_dir):
                                os.makedirs(cap_dir)

                            cap_path = cap_dir + "/{}.png".format(cap_time)

                            cv2.imwrite(cap_path, image_np)
                            print("---- {}  saved--".format(cap_path))

                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            cv2.destroyAllWindows()
                            break




        print('---------object_detection test done---------------')

        return HttpResponse("Success")



