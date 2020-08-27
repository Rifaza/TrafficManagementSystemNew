import os
from django.contrib.gis.utils import LayerMapping
from .models import Roads, Junctions, Hospitals, Normals, Schools

roads_mapping = {
    'road_name': 'Road_Name',
    'r_id': 'R_id',
    'geom': 'MULTILINESTRING25D',
}

roads_shp=os.path.abspath(os.path.join(os.path.dirname(__file__),'data/Road.shp'))


def run(verbose=True):
    lm = LayerMapping(Roads,roads_shp, roads_mapping, transform = True,encoding='iso-8859-1')
    lm.save(strict=True,verbose=verbose)


junctions_mapping = {
    'jid': 'Jid',
    'jname': 'JName',
    'geom': 'MULTIPOINT25D',

}
junctions_shp=os.path.abspath(os.path.join(os.path.dirname(__file__),'data/Junction.shp'))

def runJunction(verbose=True):
    lm = LayerMapping( Junctions,junctions_shp,  junctions_mapping, transform = True,encoding='iso-8859-1')
    lm.save(strict=True,verbose=verbose)

hospitals_mapping = {
    'h_name' : 'H_Name',
    'rid' : 'RID',
    #'fk':{'rid' : 'RID'},
    'h_id' : 'h_id',
    'region_id' : 'region_id',
    'geom' : 'MULTIPOINT25D',
}


hospitals_shp=os.path.abspath(os.path.join(os.path.dirname(__file__),'data/Hospital.shp'))

def runHospital(verbose=True):
    lm = LayerMapping( Hospitals,hospitals_shp,  hospitals_mapping, transform = True,encoding='iso-8859-1')
    lm.save(strict=True,verbose=verbose)


normals_mapping = {
    'place_name' : 'Place_Name',
    'rid' : 'RID',
    #'fk':{'rid' : 'RID'},
    'n_id' : 'n_id',
    'region_id' : 'region_id',
    #'fk':{'region_id' : 'region_id'},
    'geom' : 'MULTIPOINT25D',
}

normals_shp=os.path.abspath(os.path.join(os.path.dirname(__file__),'data/Normal.shp'))

def runNormal(verbose=True):
    lm = LayerMapping(Normals,normals_shp,normals_mapping, transform = True,encoding='iso-8859-1')
    lm.save(strict=True,verbose=verbose)
schools_mapping = {
    'schoolname' : 'SchoolName',
    'rid' : 'RID',
    #'fk':{'rid' : 'RID'},
    's_id' : 's_id',
    'region_id' : 'region_id',
    'geom' : 'MULTIPOINT25D',
}


schools_shp=os.path.abspath(os.path.join(os.path.dirname(__file__),'data/School.shp'))

def runSchool(verbose=True):
    lm = LayerMapping(Schools,schools_shp,schools_mapping, transform = True,encoding='iso-8859-1')
    lm.save(strict=True,verbose=verbose)