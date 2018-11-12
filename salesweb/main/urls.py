#-*- coding: utf-8 -*-

from django.contrib import admin
from django.conf.urls import url
from main.views import index, trainModel, manageModel, getModels, ModelCkpt, graphExport

app_name = 'main'
urlpatterns = [
	# 메인화면.
    url(r'^$', index.as_view(), name='home'),
    
    # Training.
    url(r'^trainModel', trainModel.as_view()),

    # Managing.
    url(r'^manageModel', manageModel.as_view()),

    # Testing.
    url(r'^testModel', graphExport.as_view()),

    # Get Model.
    # url(r'^getModels', getModels.as_view()),

    # Get child Modelname 
    url(r'^ModelChild$', ModelCkpt.getModelChild),

    # Get model-ckp t
    url(r'^ModelCkpt$', ModelCkpt.getModelCkpt)

]

