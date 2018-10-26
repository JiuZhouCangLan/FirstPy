#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tkinter import *
from tkinter.filedialog import askopenfilename
from imageai.Prediction import ImagePrediction
from imageai.Detection import ObjectDetection
from PIL import Image, ImageTk
from googletrans import Translator
import os

root = Tk()
root.title('AI对象识别')
root.geometry('800x700')
menubar = Menu(root)
# 模式选择菜单及变量
ProcessModeMenu = Menu(menubar, tearoff=0)
ProcessMode = StringVar()
ProcessModeName = ('图像预测', '对象检测', '视频对象检测和跟踪')
# 预测功能菜单及变量
PredictionModelMenu = Menu(menubar, tearoff=0)
PredictionModel = StringVar()
PredictionModelName = ('SqueezeNet', 'ResNet50', 'InceptionV3', 'DenseNet121')
PredictionModelPath = StringVar()
PredictionResult = StringVar()
PredictionLabel = Label(root, textvariable=PredictionResult)
PredictionSpeed = ('normal', 'fast', 'faster', 'fastest')
# 检测功能菜单及变量
DetectionMenu = Menu(menubar, tearoff=0)

imagePath = StringVar()  # 图片路径
imagelabel = Label(root)  # 图片标签组件
execution_path = os.getcwd()  # 获取当前路径

SpeedSelector = Scale(root, from_=1, to=4, orient=HORIZONTAL, label='预测速度')
SpeedSelector.grid(row=0, column=0)
CountSelector = Scale(root, from_=1, to=5, orient=HORIZONTAL, label='预测数量')
CountSelector.grid(row=0, column=1)
# 菜单选项初始化
ProcessMode.set('图像预测')
PredictionModel.set('SqueezeNet')


def prediction_model():
    global PredictionModel
    global PredictionModelPath
    if PredictionModel.get() == 'SqueezeNet':
        PredictionModelPath = os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5")
    elif PredictionModel.get() == 'ResNet50':
        PredictionModelPath = os.path.join(execution_path, "resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    elif PredictionModel.get() == 'InceptionV3':
        PredictionModelPath = os.path.join(execution_path, "inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    elif PredictionModel.get() == 'DenseNet121':
        PredictionModelPath = os.path.join(execution_path, "DenseNet-BC-121-32.h5")
    return PredictionModelPath


def detectionmode():
    global ProcessMode
    ProcessMode.set('对象检测')


ProcessModeMenu.add_radiobutton(label='图像预测', value='图像预测', variable=ProcessMode)
ProcessModeMenu.add_radiobutton(label='对象检测', value='对象检测', variable=ProcessMode)
menubar.add_cascade(label='功能选择', menu=ProcessModeMenu)
for v in PredictionModelName:
    PredictionModelMenu.add_radiobutton(label=v, command=prediction_model(), value=v, variable=PredictionModel)
menubar.add_cascade(label='图像预测模型', menu=PredictionModelMenu)


def selectimage():
    global imagePath
    global imagelabel
    global PredictionResult
    path_ = askopenfilename(filetypes=[("图片文件", "*.jpg;*.bmp;*.png")])
    print(path_)
    if path_ != "":
        imagePath = path_
        img_open = Image.open(imagePath)
        h, w = img_open.size
        img_open.thumbnail((720, 720*h/w))
        img = ImageTk.PhotoImage(img_open)
        imagelabel.config(image=img)
        imagelabel.image = img
        PredictionResult.set('')
    else:
        print("未选择文件")


menubar.add_command(label='选择图片', command=selectimage)


def zh_cn(source):
    translator = Translator(service_urls=['translate.google.cn'])
    text = translator.translate(source, dest='zh-cn').text
    if source != text:
        return text
    else:
        return ""


def process():
    global imagePath
    global ProcessMode
    global PredictionSpeed
    speedindex = SpeedSelector.get()
    if ProcessMode.get() == '图像预测':
        global PredictionModelPath
        global PredictionResult
        prediction = ImagePrediction()
        PredictionResult.set('')
        if PredictionModel.get() == 'SqueezeNet':
            print('预测模型选中：SqueezeNet')
            prediction.setModelTypeAsSqueezeNet()
        elif PredictionModel.get() == 'ResNet50':
            print('预测模型选中：ResNet50')
            prediction.setModelTypeAsResNet()
        elif PredictionModel.get() == 'InceptionV3':
            print('预测模型选中：InceptionV3')
            prediction.setModelTypeAsInceptionV3()
        elif PredictionModel.get() == 'DenseNet121':
            print('预测模型选中：DenseNet121')
            prediction.setModelTypeAsDenseNet()
        PredictionModelPath = prediction_model()
        print('模型路径：' + PredictionModelPath)
        prediction.setModelPath(PredictionModelPath)
        print('识别速度'+PredictionSpeed[speedindex-1])
        prediction.loadModel(prediction_speed=PredictionSpeed[speedindex-1])
        predictions, probabilities = prediction.predictImage(imagePath, result_count=CountSelector.get())
        for eachPrediction, eachProbability in zip(predictions, probabilities):
            PredictionResult.set(PredictionResult.get() + "\n" +
                                 str(eachPrediction) + zh_cn(str(eachPrediction)) + " : " + str(eachProbability))
    elif ProcessMode.get() == '对象检测':
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
        detector.loadModel()
        detections = detector.detectObjectsFromImage(input_image=imagePath,
                                                     output_image_path=os.path.join(execution_path, "cache.jpg"))
        PredictionResult.set('')
        for eachObject in detections:
            PredictionResult.set(PredictionResult.get()+str(eachObject["name"]) + zh_cn(str(eachObject["name"]))
                                 + " : " + str(eachObject["percentage_probability"])+"\n")
        img_open = Image.open(os.path.join(execution_path, "cache.jpg"))
        h, w = img_open.size
        img_open.thumbnail((720, 720 * h / w))
        img = ImageTk.PhotoImage(img_open)
        imagelabel.config(image=img)
        imagelabel.image = img
    elif ProcessMode.get() == '视频对象检测和跟踪':
        print('视频对象检测和跟踪功能占位')


btn_process = Button(root, text='执行', width=15, height=2, command=process)
root['menu'] = menubar
btn_process.grid(row=1, column=0, columnspan=2)
imagelabel.grid(row=2, column=0, columnspan=2)
PredictionLabel.grid(row=3, column=0, columnspan=2)
root.mainloop()
