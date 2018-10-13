#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tkinter import *
from tkinter.filedialog import askopenfilename
from imageai.Prediction import ImagePrediction
from PIL import Image, ImageTk
import os
import time

root = Tk()
root.title('AI识别')
root.geometry('800x600')
menubar = Menu(root)
modelselect = Menu(menubar, tearoff=0)
model = StringVar()
modelname = ('SqueezeNet', 'ResNet50', 'InceptionV3', 'DenseNet121')
imagePath = StringVar()
imagelable = Label(root)

execution_path = os.getcwd()  # 模型文件
prediction = ImagePrediction()  # 创建预测类


def setmodelpath():
    global model
    if model.get() == 'SqueezeNet':
        modelpath = os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5")
    elif model.get() == 'ResNet50':
        modelpath = os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    elif model.get() == 'InceptionV3':
        modelpath = os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    elif model.get() == 'DenseNet121':
        modelpath = os.path.join(execution_path, "DenseNet-BC-121-32.h5")
    return modelpath


def selectimage():
    global imagePath
    path_ = askopenfilename(filetypes=[("图片文件", "*.jpg;*.bmp;*.png")])
    imagePath = path_
    img_open = Image.open(imagePath)
    h, w = img_open.size
    img_open.thumbnail((720, 720*h/w))
    img = ImageTk.PhotoImage(img_open)
    imagelable.config(image=img)
    imagelable.image = img


for v in modelname:
    modelselect.add_radiobutton(label=v, command=setmodelpath, variable=model)
menubar.add_cascade(label='模型', menu=modelselect)
menubar.add_command(label='选择图片', command=selectimage)


def recognition():
    global imagePath
    modelpath = setmodelpath()
    print(imagePath)
    print(modelpath)
    start = time.time()
    prediction.setModelTypeAsDenseNet()
    prediction.setModelPath(modelpath)
    prediction.loadModel()
    predictions, probabilities = prediction.predictImage(imagePath, result_count=5)
    end = time.time()
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(str(eachPrediction) + " : " + str(eachProbability))

    print("\ncost time:", end - start)


btn_recognition = Button(root, text='识别', width=15, height=2, command=recognition)
root['menu'] = menubar
btn_recognition.pack()
imagelable.pack()
root.mainloop()
