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
	methodname     = models.CharField(max_length=30, null=False, default='SSD')
	modelname  = models.CharField(max_length=100, null=False, blank=False, default='model_blank')
	datetime   = models.CharField(max_length=12, blank=False)
	filename   = models.CharField(max_length=100)
	label      = models.CharField(max_length=100)
	px         = models.IntegerField(default=0)
	py         = models.IntegerField(default=0)
	width      = models.IntegerField(default=0)
	height     = models.IntegerField(default=0)
	image_path = models.TextField()
	hit_yn     = models.CharField(max_length=1, default="Y", blank=False)
	ip         = models.CharField(max_length=15)

	def __str__(self):
		return '{}/{}/{}/{}'.format(self.methodname , self.modelname , self.datetime, self.filename)

	def get_path(self):
		return self.image_path

