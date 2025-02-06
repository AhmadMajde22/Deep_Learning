from django.shortcuts import render
from .forms import FingerUploadForm
from django.conf import settings
import os
import numpy as np
from PIL import Image
import tensorflow as tf


model = tf.keras.models. load_model('/mnt/e/CODES/WSL_JUPYTER/fingers/model.keras')

def predict_fingers(image_path):
    image = Image.open(image_path).resize((224,224)).convert('RGB')
    image_array  = np.array(image)/255.0
    image_array = np.expand_dims(image_array,axis = 0)

    predictions = model.predict(image_array,verbose = 0)
    pred = np.argmax(predictions,axis = 1)

    labels = {0: 'Zero fingers', 1: 'One finger', 2: 'Two fingers', 3 : 'Three fingers', 4: 'Four fingers', 5 : 'Five fingers'}
    predicted_label = labels.get(pred[0], "Unknown")

    return predicted_label

def index(request):
    if request.method == 'POST':
        form = FingerUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_name = f"finger_image_{image.name}"
            image_path = os.path.join(settings.MEDIA_ROOT, image_name)

            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            prediction = predict_fingers(image_path)
            image_url = os.path.join(settings.MEDIA_URL, image_name)

            return render(request, 'Finger_app/result.html', {
                'prediction': prediction,
                'image_url': image_url
            })
    else:
        form = FingerUploadForm()
    return render(request, 'Finger_app/index.html', {'form': form})
