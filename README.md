>This is the project of the StereoMatching Project. This project based on my framework (if you want to use it to build the Network, you can find it on my website: [fadeshine](http://www.fadeshine.com/). If you have any questions, you can send an e-mail to me. My e-mail: raoxi36@foxmail.com)

### Software Environment
1. OS Environment
os >= linux 16.04
cudaToolKit == 10.1
cudnn == 7.3.6
2. Python Environment
python == 3.8.5
pythorch == 1.15.0
numpy == 1.14.5
opencv == 3.4.0
PIL == 5.1.0

### Hardware Environment
Training process:
- GPU: 1080TI * 4 or other memory at least 11G.(Batch size: 1)
if you not have four gpus, you could change the para of model. The Minimum Testing process:
- GPU: memory at least 5G. (Batch size: 1)

### Train the model by running:
0. Install the JackFramework lib from Github (https://github.com/Archaic-Atom/JackFramework)
```
$ cd JackFramework/
$ ./install.sh
```

1. Get the Training list or Testing list （You need rewrite the code by your path, and my related code can be found in Source/Tools）
```
$ ./GenPath.sh
```
Please check the path. The source code in Source/Tools.

2. Run the pre-training process (This is pre-training process. We will provide the pre-trained model at BaiduYun or Google Driver)
```
$ ./Scripts/start_train_scene_flow_stereo_net.sh
```
Please carefully check the path in related file.

3. Run the training cmd (This is fine-tuing process.bg means background running. note that please check the img path should be found in related path, e.g. ./Dataset/trainlist_ETH3D.txt)
```
$ ./TrainKitti_2012_bg.sh
or
$ ./TrainKitti_2015_bg.sh
or
$ ./TrainKitti_ROB_bg.sh
```
Please carefully check the path in related file.

4. Run the testing cmd
```
$ ./Scripts/start_test_kitti2012_stereo_net.sh
or 
$ ./Scripts/start_test_kitti2015_stereo_net.sh.sh
or 
$ ./Scripts/start_test_eth3d_stereo_net.sh.sh
or 
$ ./Scripts/start_test_middlebury_stereo_net.sh (for test data)
or 
$ ./Scripts/start_test_sceneflow_stereo_net.sh
```

if you want to change the para of the model, you can change the *.sh file. Such as:
```
$ vi ./Scripts/start_test_kitti2012_stereo_net.sh
or 
$ vi ./Scripts/start_test_eth3d_stereo_net.sh.sh
```

### File Structure
```
.
├── Source # source code
│   ├── UserModelImplementation
│   ├── Tools
│   ├── main.py
│   └── ...
├── Datasets # Get it by ./GenPath.sh, you need build folder
│   ├── kitti2012_val_list.csv.txt
│   ├── kitti2015_val_list.csv.txt
│   └── ...
├── Result # The data of Project. Auto Bulid
│   ├── output.log
│   ├── train_acc.csv
│   └── ...
├── ResultImg # The image of Result. Auto Bulid
│   ├── 000001_10.png
│   ├── 000002_10.png
│   └── ...
├── Checkpoints # The saved model. Auto Bulid
│   ├── checkpoint
│   └── ...
├── log # The graph of model. Auto Bulid
│   ├── events.out.tfevents.1541751559.ubuntu
│   └── ...
├── Scripts # shell cmd
│   ├──GetPath.sh
│   ├──Pre-Train.sh
│   └── ...
├── LICENSE
├── requirements.txt
└── README.md
```

### Update log
#### 2021-05-29
1. Add the depth for transformer;
2. Fork the JackFramework to a new project;
3. Remove the JackFramework from this project.

#### 2021-04-08
1. Add the stereo;
2. Add transformer.

#### 2021-01-13
1. Fork a new prject (based on pythorch);
2. Use a new code style;
3. Build the frameworks for pythorch;
4. Write ReadMe