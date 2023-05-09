# computer-vision-visionaries
# Soccer Tracking

## Instructions for local setup.

If you wish to train the network, or run inference on larger videos, we recommend running locally on WSL, as it eases many compatibility issues especially with regard to GPU acceleration in Torch/TF. We have not had success running on Windows native. 

To download this, run:
```bash
git clone https://github.com/henrypdonahue/computer-vision-visionaries
```

Requirements: 
- Conda w/ Python 3.7, 3.8, or 3.9 (might work with more updated versions but have not been tested)
    - We recommend a conda environment
- CUDA-enabled GPU (recommended > 8gb VRAM)

## Installing YOLOv5
- Download yolov5 by running
```bash
git clone https://github.com/ultralytics/yolov5
```
Then install dependencies: 
```bash
pip install -r yolov5/requirements.txt
```

## Downloading Kaggle Bundesliga data.
While this dataset pertains to a separate problem, so it was not used to train, it does have high quality 
videos that can be used for inference/demos.

- Latest version of Kaggle library:
```bash
pip install kaggle
```  
- Create Kaggle username and get a valid API key
- Add username/key to setup.py parameters

- Change the kaggle path in setup.py to the path of kaggle.json
    - If you do not have this run:
        ```bash
        mkdir ~/.kaggle 
        ```
        ```bash
        touch ~/.kaggle/kaggle.json 
        ```

- Now run setup.py
    ```bash 
    python Soccer-Tracking/setup.py
    ```

- If you get some permission error run, and try again:
    ```bash 
    chmod 600 ~/.kaggle/kaggle.json
    ```

- To download bundesliga dataset run:
    ```bash 
    kaggle competitions download -c dfl-bundesliga-data-shootout
    ```
    - Note that this is a very large file (about 34 gb), so you can just download a few if you
    just want to test with a pre-trained model
    - Unzip the download: 
        ```bash 
        unzip dfl-bundesliga-data-shootout.zip -d bundesliga-data
        ```

## Using pre-trained YOLO weights
- Choose a model:
    - YOLO has many different pre-trained models of different sizes and speeds that can be used: https://github.com/ultralytics/yolov5#pretrained-checkpoints
    - We've had the most success using the 1280 pixel weights, especially with regard to soccer ball recognition.
    - If only being used for short inference, the largest possible model should be used, as 30 second clips should only 
    take about a minute to run on a decent GPU, even with the largest models.
- Then run, optionally changing the weights (can be chosen from pretrained checkpoints list), img (image size, usually 640 or 1280), conf (minimum confidence to show prediction usually between .25 and .6), source (path to an image or mp4), and project (name for desired directory for outputs) parameters:
```bash
python yolov5/detect.py --weights yolov5/yolov5x6.pt --img 1280 --conf 0.25 --source PATH_TO_BUNDESLIGA_DATA/clips/08fd33_4.mp4 --project DFL
```
- It seems like are some problems regarding WSL2 and a recent version of Ubuntu and recognizing CUDA
    - If you run into this try running:
    ```bash
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    ```
    - Then verify with: 
    ```bash
    ldconfig -p | grep cuda
    ```
    - This should also be added to the bashrc if you don't want to run it every time you open bash
    - Source: https://discuss.pytorch.org/t/libcudnn-cnn-infer-so-8-library-can-not-found/164661

## Using our trained weights
Download and unzip our weights from: https://drive.google.com/file/d/1L3DSgSJ-t_yYDvOrCKj0o4EQTWtkjZ2Z/view?usp=share_link (need Brown email to access). 
Example call: 
```bash
python yolov5/detect.py --weights yolov5/best_m6_100ep.pt --img 1280 --conf 0.25 --source PATH_TO_BUNDESLIGA_DATA/clips/08fd33_4.mp4 --project DFL
```
## Using specialized ball detection weights
- From: https://www.kaggle.com/code/shinmurashinmura/dfl-yolov5-ball-detection/input?select=yolov5l6_trained_600images.pt
Example call: 
```bash
python yolov5/detect.py --weights yolov5/yolov5l6_trained_600images.pt --img 1280 --conf 0.2 --source PATH_TO_BUNDESLIGA_DATA/clips/08fd33_4.mp4 --project DFL
```

## Training your own weights
Training weights for this problem is relatively difficult due to the overall lack of quality data. For future projects, this is likely the area 
with largest room for improvement with better data or more compute power. A good place to start may be the SoccerNet dataset, which, while it uses 
a fairly different data format, it should be possible to convert into the expected YOLOv5 data format, described here: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#11-create-datasetyaml. 

We used this dataset to train:
https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc

Example call:
python yolov5/train.py --img 1280 --batch 6 --epochs 100 --data data.yaml --weights yolov5m6.pt
(Our best model for around 12gb VRAM)

Some important considerations/tips based on our experience training:
- weights
    - The YOLOv5 train script allows to have pre-trained weights as a starting point for the model. Given the scarcity of data, we highly recommend this. 
    - Any of the pretrained checkpoints provided by ultralytics can be used: https://github.com/ultralytics/yolov5#pretrained-checkpoints, or our provided 
    pretrained weights
    - The number of parameters for each model have a significant impact on the training performance, and can limit the batch size depending on your VRAM
    budget.
    - We've found the the 'm' or 'l' models probably have the best memory/speed to performance trade-off for this dataset.
    - The models ending in 6, which train on a default image size of 1280, performed best for this dataset. 
- batch
    - The batch size also has a large impact on training performance, and is largely limited by the GPU's VRAM budget.
    - In general, ultralytics recommends using as large of a batch size as possible in your system for the model size, and 
    we found this to be true for this dataset. 
    - If you get a killed or aborted error, the model likely ran out of memory and the batch size or model size should be decreased. Note that 
    decreasing either of these parameters causes underfitting, so it is important to manage trade-offs and ajust both of them accordingly.
- epochs
    - In general, we've found YOLO's training method to be very robust and resistant to overfitting, so train for as many epochs as 
    you are willing to wait. 
- cache
    - Ultralytics recommends to set this option to --cache ram or --cache disk to increase performance, but this caused some trouble in our 
    system so we ultimately omitted this option.

## Integrating with ByteTrack
The provided notebook is adapted from this tutorial: https://blog.roboflow.com/track-football-players/, with some changes such as using separate player/ref/goalie and ball detection, visualization changes, etc. 

You can download our provided weights, from [Using our trained weights](#using-our-trained-weights), as well as ball-specific weights, from [Using specialized ball detection weights](#using-specialized-ball-detection-weights)
