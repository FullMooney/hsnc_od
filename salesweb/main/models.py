from __future__ import unicode_literals
from datetime import datetime
from django.db import models
from django.conf import settings



# Create your models here.
class TrainingModel(models.Model):
    modelname = models.CharField(max_length=100, null=True, blank=True)
    pid       = models.CharField(max_length=100, null=True, blank=True)
    label     = models.TextField()
    method    = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(default=datetime.now, blank=True)

    def __str__(self):
        return self.modelname


class ResultModel(models.Model):
	"""docstring for TestingModel"""
	
	datetime   = models.TextField(blank=False)
	seq        = models.IntegerField(default=1) 
	label      = models.TextField()
	px         = models.IntegerField(default=0)
	py         = models.IntegerField(default=0)
	width      = models.IntegerField()
	height     = models.IntegerField()
	image_path = models.CharField(max_length=100)
	hit_yn     = models.CharField(max_length=1, default="Y", blank=False)
	ip         = models.TextField() 

	def __str__(self):
		return self.datetime

	def get_path(self):
		return self.image_path

