# Generated by Django 2.1.2 on 2018-11-19 14:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_auto_20181010_1943'),
    ]

    operations = [
        migrations.CreateModel(
            name='ResultModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('methodname', models.CharField(default='SSD', max_length=30)),
                ('modelname', models.CharField(default='model_blank', max_length=100)),
                ('datetime', models.CharField(max_length=12)),
                ('filename', models.CharField(max_length=100)),
                ('label', models.CharField(max_length=100)),
                ('px', models.IntegerField(default=0)),
                ('py', models.IntegerField(default=0)),
                ('width', models.IntegerField(default=0)),
                ('height', models.IntegerField(default=0)),
                ('image_path', models.TextField()),
                ('hit_yn', models.CharField(default='Y', max_length=1)),
                ('ip', models.CharField(max_length=15)),
            ],
        ),
    ]
