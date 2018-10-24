# -*- coding: utf-8 -*-

from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound
from django.template import RequestContext
from django.views.generic import View, TemplateView
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