import gradio as gr
from tensorflow import  keras
import numpy as np
model = keras.models.load_model('digit_recong.h5')
def recognize_digit(img):
    img=np.array(img)
    img=img.reshape([1,28,28,1])
    prediction = model.predict(img)
    pred=np.argmax(prediction[0])

    return str(pred)
label=gr.outputs.Label(type='auto')
gr.Interface(fn=recognize_digit, inputs="sketchpad", outputs=label).launch(share=True)