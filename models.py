from __future__ import unicode_literals
from django.db import models
from django.contrib.gis.db import models as gismodels
from django.db.models import Manager as GeoManager

# Create your models here.

class Cons(models.Model):
    
    region=models.IntegerField(max_length=20)
    time_gmt=models.IntegerField()
    wday=models.IntegerField()
    o3=models.FloatField()
    pm25=models.FloatField()
    no2=models.FloatField()
    place_id=models.CharField(max_length=10)

    # def __unicode__(self):
    #     return self.id
    
    class Meta:
        verbose_name_plural="Cons"
class StartPlaces(models.Model):
    name=models.CharField(max_length=20)
    location=gismodels.PointField(srid=4326)
    objects = GeoManager()

    def __unicode__(self):
        return self.name
    
    class Meta:
        verbose_name_plural="StartPlaces"
class EndPlaces(models.Model):
    name=models.CharField(max_length=20)
    location= gismodels.PointField(srid=4326)
    objects = GeoManager()
    def __unicode__(self):
        return self.name
    
    class Meta:
        verbose_name_plural="Distinations"


class Regions(models.Model):
    region_id = models.IntegerField()
    region_name=models.CharField(max_length=20)

    def __unicode__(self):
        return self.region_name
    
    class Meta:
        verbose_name_plural="Regions"



class Roads(models.Model):
    road_name = models.CharField(max_length=30)
    r_id = models.IntegerField()
    geom = gismodels.MultiLineStringField(srid=4326)

    objects = GeoManager()
    #objects = gismodels.GeoManager()

    def __unicode__(self):
        return self.road_name
    
    
    class Meta:
        verbose_name_plural="Roads"
    

class Junctions(models.Model):
    jid = models.CharField(max_length=15)
    jname = models.CharField(max_length=20)
    geom = gismodels.MultiPointField(srid=4326)
    objects =GeoManager()
    roads=models.ManyToManyField(Roads)


    def __unicode__(self):
        return self.j_name
    
    
    class Meta:
        verbose_name_plural="Junctions"



class Hospitals(models.Model):
    h_name = models.CharField(max_length=80)
    rid=models.IntegerField()
    #rid = models.ForeignKey(Roads, on_delete=models.CASCADE)
    h_id = models.CharField(max_length=15)
    region_id = models.IntegerField()

    #region_id=models.ForeignKey(Regions, on_delete=models.CASCADE)
    geom = gismodels.MultiPointField(srid=4326)

    



    def __unicode__(self):
        return self.h_name
    
    
    class Meta:
        verbose_name_plural="Hospitals"
class Normals(models.Model):

    place_name = models.CharField(max_length=40)
    rid=models.IntegerField()
    #rid = models.ForeignKey(Roads, on_delete=models.CASCADE)
    n_id = models.CharField(max_length=15)
    region_id = models.IntegerField()

    #region_id=models.ForeignKey(Regions, on_delete=models.CASCADE)
    geom = gismodels.MultiPointField(srid=4326)
    objects =GeoManager()



    def __unicode__(self):
        return self. place_name
    
    
    class Meta:
        verbose_name_plural="Normals"


class Schools(models.Model):
    schoolname = models.CharField(max_length=25)
    rid=models.IntegerField()
    #rid = models.ForeignKey(Roads, on_delete=models.CASCADE)
    s_id = models.CharField(max_length=15)
    region_id = models.IntegerField()
    #region_id=models.ForeignKey(Regions, on_delete=models.CASCADE)
    geom =gismodels.MultiPointField(srid=4326)
    objects =GeoManager()



    def __unicode__(self):
        return self.schoolname
    
    
    class Meta:
        verbose_name_plural="Normals"





