# Generated by Django 3.1.1 on 2021-01-11 13:25

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('app_yolov5_api', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='pois_map',
            name='picture',
            field=models.TextField(blank=True),
        ),
        migrations.AlterField(
            model_name='pois_map',
            name='created_at',
            field=models.DateTimeField(default=datetime.datetime(2021, 1, 11, 21, 25, 0, 566384, tzinfo=utc)),
        ),
    ]