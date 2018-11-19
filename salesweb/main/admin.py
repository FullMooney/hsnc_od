from django.contrib import admin

from .models import TrainingModel, ResultModel

# Register your models here.

admin.site.register(TrainingModel)

admin.site.register(ResultModel)