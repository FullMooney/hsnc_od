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