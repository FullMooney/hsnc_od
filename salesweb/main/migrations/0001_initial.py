# Generated by Django 2.1.1 on 2018-10-10 19:28

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TrainModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('modelname', models.CharField(blank=True, max_length=100, null=True)),
                ('pid', models.CharField(blank=True, max_length=100, null=True)),
                ('label', models.TextField()),
                ('method', models.CharField(blank=True, max_length=100, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
