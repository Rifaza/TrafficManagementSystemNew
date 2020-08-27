from django.contrib import admin

from leaflet.admin import LeafletGeoAdmin
from .models import  StartPlaces, Roads,Junctions, Hospitals, Normals,EndPlaces,Regions,Cons


class  StartPlacesAdmin(LeafletGeoAdmin):
    
    list_display=('name','location')

admin.site.register(  StartPlaces,  StartPlacesAdmin)

class  RegionsAdmin(LeafletGeoAdmin):
    
    list_display=('region_id','region_name')

admin.site.register(  Regions,  RegionsAdmin)
# Register your models here.
class  RoadsAdmin(LeafletGeoAdmin):
    
    list_display=('road_name','r_id','geom')

admin.site.register(  Roads,  RoadsAdmin)
class  JunctionsAdmin(LeafletGeoAdmin):
    
    list_display=('jid','jname','geom')

admin.site.register(  Junctions, JunctionsAdmin)


class  HospitalsAdmin(LeafletGeoAdmin):
    
    list_display=('id','h_name','geom')

admin.site.register(  Hospitals,  HospitalsAdmin)


class  NormalsAdmin(LeafletGeoAdmin):
    
    list_display=('id','place_name','rid','geom')

admin.site.register(  Normals,   NormalsAdmin)
class  EndPlacesAdmin(LeafletGeoAdmin):
    
    list_display=('name','location')

admin.site.register(  EndPlaces,  EndPlacesAdmin)

class  ConsAdmin(LeafletGeoAdmin):
    
    list_display=('id','region','time_gmt','wday','o3','pm25','no2','place_id')

admin.site.register(  Cons,  ConsAdmin)