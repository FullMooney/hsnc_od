# -*- coding: utf-8 -*-

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.template import RequestContext
from django.views.generic import View, TemplateView, DetailView
from decimal import Decimal
from .models import TrainingModel
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

## model selection
from django.db.models import Q
from django.core import serializers

from os import listdir
from os.path import isfile, join
from shutil import copyfile, copy2, rmtree



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
        print(outarr)
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
            fpathList += filepath + ','
            rpathList += rpath + ','
            resultpath.append('{}/{}'.format(staticImgPath, file.name))
            print(filepath)
            

        args = [child, fpathList, rpathList] 
        print(args)
        
        testProc = subprocess.Popen(["python", BASE_DIR + '/main/object_detection/object_detection_test.py'] + args , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # print(os.getcwd() + '/main/object_detection/object_detection_test.py')
        output, stderr = testProc.communicate()
        outarr = output.decode('utf-8').splitlines()

        # delete temp folder
        rmtree(fpath)
        print('---------object_detection test done---------------')

        return HttpResponse(json.dumps(resultpath), content_type='application/json')   

    def get_image_url():
        pass