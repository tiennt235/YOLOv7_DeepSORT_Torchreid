import argparse
import time
from pathlib import Path

import sys
import os
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from deep_sort.application_util import preprocessing
from deep_sort.application_util import visualization
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet

# reid
from reid import REID
import operator

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace, reid = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.reid
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # DeepSORT: init tracker
    max_cosine_distance = 0.4
    nn_budget = None
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    
    # DeepSORT: init re-ID encoder
    model_filename = './model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    # reid
    all_frames = []
    track_cnt = dict()
    frames_by_id = dict()
    ids_per_frame = []
    
    t0 = time.time()
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        tmp_ids = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
            # reid
            all_frames.append(im0)
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # get bounding boxes for DeepSORT
                xyxy = det[:, :4].cpu()
                # create top-left-width-height bounding box
                tlwhs =  xyxy[:, :4].clone() if isinstance(xyxy, torch.Tensor) else np.copy(xyxy[:, :4])
                tlwhs[:, 0] = xyxy[:, 0] #  center
                tlwhs[:, 1] = xyxy[:, 1] # y center
                tlwhs[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width
                tlwhs[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height
                
                confs = det[:, 4]
                clss = det[:, 5]
                
                # DeepSORT: get re-ID feature
                features = encoder(im0, tlwhs)
                
                # create detections to feed to DeepSORT
                detections = [Detection(tlwh, conf, feature) for tlwh, conf, feature in zip(tlwhs, confs, features)]
                
                # DeepSORT: update detections 
                t4 = time_synchronized()
                tracker.predict()
                tracker.update(detections)      
                t5 = time_synchronized()
                  
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # DeepSORT: store result
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    bbox = track.to_tlbr()
                    track_id = track.track_id
                    cls = 0
                    
                    # reid
                    tmp_ids.append(track_id)
                    area = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
                    if track_id not in track_cnt:
                        track_cnt[track_id] = [
                            [frame_idx, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), area]
                        ]
                        frames_by_id[track_id] = [im0[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                    else:
                        track_cnt[track_id].append([
                            frame_idx,
                            int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                            area
                        ])
                        frames_by_id[track_id].append(im0[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                        
                        
                    if save_txt:
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, track_id, bbox[0], bbox[1], bbox[2], bbox[3] -1, -1, -1, i))
                    
                    # DeepSORT: plot track
                    if save_img:
                        c = int(cls)  # integer class
                        track_id = int(track_id)  # integer id
                        label = f'{names[c]} {track_id}'
                        plot_one_box(bbox, im0, label=label, color=colors[int(track_id)], line_thickness=2)
                        
                # reid
                ids_per_frame.append(set(tmp_ids))
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{names[int(cls)]} {conf:.2f}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS, ({(1E3 * (t5 - t4)):.1f}ms) DeepSORT')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    
    # Reid post-processing
    if reid:
        reid = REID()
        threshold = 320
        exist_ids = set()
        final_fuse_id = dict()

        print(f'Total IDs = {len(frames_by_id)}')
        feats = dict()
        for i in frames_by_id:
            print(f'ID number {i} -> Number of frames {len(frames_by_id[i])}')
            feats[i] = reid._features(frames_by_id[i])  # reid._features(frames_by_id[i][:min(len(frames_by_id[i]),100)])

        for f in ids_per_frame:
            if f:
                if len(exist_ids) == 0:
                    for i in f:
                        final_fuse_id[i] = [i]
                    exist_ids = exist_ids or f
                else:
                    new_ids = f - exist_ids
                    for nid in new_ids:
                        dis = []
                        if len(frames_by_id[nid]) < 10:
                            exist_ids.add(nid)
                            continue
                        unpickable = []
                        for i in f:
                            for key, item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                        for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                            tmp = np.mean(reid.compute_distance(feats[nid], feats[oid]))
                            print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                            dis.append([oid, tmp])
                        exist_ids.add(nid)
                        if not dis:
                            final_fuse_id[nid] = [nid]
                            continue
                        dis.sort(key=operator.itemgetter(1))
                        if dis[0][1] < threshold:
                            combined_id = dis[0][0]
                            frames_by_id[combined_id] += frames_by_id[nid]
                            final_fuse_id[combined_id].append(nid)
                        else:
                            final_fuse_id[nid] = [nid]
        print('Final ids and their sub-ids:', final_fuse_id)
        print('MOT took {} seconds'.format(int(time.time() - t1)))
        
        vid_cap = cv2.VideoCapture(source)
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(str(save_dir / 'reid.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        for frame_idx, frame in enumerate(all_frames):
            for idx in final_fuse_id:
                for i in final_fuse_id[idx]:
                    for f in track_cnt[i]:
                        # print('frame {} f0 {}'.format(frame,f[0]))
                        if frame_idx == f[0]:
                            label = f'{names[0]} {idx}'
                            plot_one_box([f[1], f[2], f[3], f[4]], frame, label=label, color=colors[int(idx)], line_thickness=2)
            vid_writer.write(frame)
        vid_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--reid', action='store_true', help='don`t trace model')
    
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
