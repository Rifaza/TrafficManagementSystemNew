from django import forms
from .models import StartPlaces,EndPlaces,Junctions

jun=Junctions.objects.all()
for i in range(jun.count()):
    val=Junctions.objects.filter(pk__exact=i)
    GEEKS_CHOICES={i,val}
print(val)
#
# class JunctionForm(forms.Form):
#     StartPlace = forms.ChoiceField(choices=GEEKS_CHOICES)

# geek=(
#     ('0'0),
#
# )
class JunctionForm(forms.Form):
    class  Meta:
        model= Junctions
        fields=[
            'jid',
            'jname'
        ]

    jname=forms.CharField()