# Generated by Django 2.1.1 on 2018-10-10 19:43

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0003_trainingmodel'),
    ]

    operations = [
        migrations.AlterField(
            model_name='trainingmodel',
            name='created_at',
            field=models.DateTimeField(blank=True, default=datetime.datetime.now),
        ),
    ]