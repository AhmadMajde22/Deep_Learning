from django import forms

class FingerUploadForm(forms.Form):
    image = forms.ImageField()
