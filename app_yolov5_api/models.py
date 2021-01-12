from django.db import models
from datetime import datetime
from django.utils.timezone import utc
# Create your models here.

class Pois_map(models.Model):
    id = models.IntegerField(primary_key=True, auto_created=True)
    title = models.TextField(blank=False)
    point = models.TextField(blank=False)
    description = models.TextField(blank=True)
    picture = models.TextField(blank=True)
    created_at = models.DateTimeField(default=datetime.now().replace(tzinfo=utc))