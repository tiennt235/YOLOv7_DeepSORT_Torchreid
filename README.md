# YOLOv7_DeepSORT_Torchreid
Tracking using YOLOv7 with DeepSORT and Torchreid

**To run on Colab:**
```
!git clone --recurse-submodules https://github.com/tiennt235/YOLOv7_DeepSORT_Torchreid.git
!cd YOLOv7_DeepSORT_Torchreid
!pip install -r requirements.txt
```
Run yolov7/detect.py:  
```
!cd yolov7
!python detect.py --weights yolov7.pt
```
Main detection:
```
!cd path_to_YOLOv7_DeepSORT_Torchreid_ROOT
!python detect.py --weights yolov7/yolov7.pt --conf 0.5 --source /content/drive/MyDrive/CS331/videos/allVideos.avi --classes 0
```