import torch
from pygame.time import Clock

import articulate as art
import os
from config import *
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.bullet.view_rotation_np import RotationViewer
from auxiliary import calibrate_q, quaternion_inverse
from utils.model_utils import load_model
import numpy as np
import matplotlib
from argparse import ArgumentParser
import keyboard
from sensor_apple.sensor import AppleSensor, CalibratedAppleSensor
from sensor_huawei.sensor import HuaweiSensor, CalibratedHuaweiSensor
from sensor_android.sensor import AndroidSensor, CalibratedAndroidSensor
from scipy.spatial.transform import Rotation as R
from articulate.utils.noitom.PN_lab import CalibratedIMUSet
import keyboard
import traceback
import datetime

colors = matplotlib.colormaps['tab10'].colors
body_model = art.ParametricModel(paths.smpl_file, device='cuda')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--noitom', action='store_true', help='use noitom imu')
    parser.add_argument('--apple', action='store_true', help='use apple imu')
    parser.add_argument('--huawei', action='store_true', help='use huawei imu')
    parser.add_argument('--android', action='store_true', help='use android imu')
    parser.add_argument('--mocap', action='store_true', help='use mocap')
    parser.add_argument('--sviewer', action='store_true', help='show streaming data viewer')
    parser.add_argument('--device_idx', type=int, default=3, help='device index to visualize')
    args = parser.parse_args()
    
    device = torch.device("cuda")
    clock = Clock()
    
    # set mobileposer network
    if args.mocap:
        ckpt_path = "data/checkpoints/base_model.pth"
        net = load_model(ckpt_path)
        net.eval()
        print('Mobileposer model loaded.')

    if args.noitom:
        # add ground truth readings
        imu_set = CalibratedIMUSet()
        imu_set.calibrate('walking_9dof')
    
    if args.apple:
        sensor = CalibratedAppleSensor(AppleDevices.udp_ports, AppleDevices.device_ids)
        sensor.calibrate("walking_6dof")

    if args.huawei:
        sensor = CalibratedHuaweiSensor(HuaweiDevices.device_ids)
        sensor.calibrate("walking_6dof")
    
    if args.android:
        sensor = CalibratedAndroidSensor(AndroidDevices.device_ids)
        sensor.calibrate("walking_6dof")

    accs, oris, poses, trans = [], [], [], []
    
    ids = sensor.ids
    print(f"Using device IDs: {ids}")
    
    idx = 0
    with torch.no_grad(), MotionViewer(1, overlap=False, names=['every_mocap']) as viewer:
        while True:
            try:
                clock.tick(30)
                ori = torch.zeros(5, 3, 3).to(device)
                a   = torch.zeros(5, 3).to(device)
                    
                # device readings
                t, aS, aI, aM, RIS, RMB = sensor.get()
                ori[ids] = RMB.to(device)
                a[ids]   = aM.to(device)

                oris.append(ori.clone())
                accs.append(a.clone())

                if args.mocap:
                    ori = ori.view(5, 3, 3)
                    a = a.view(5, 3)

                    a = a / amass.acc_scale
                    
                    input = torch.cat([a.flatten(), ori.flatten()], dim=0).to("cuda")

                    pose = net.forward_frame(input)

                    poses.append(pose)
                    
                    pose = pose.cpu().numpy()      
                    
                    zero_tran = np.array([0, 0, 0])  
                    viewer.update_all([pose], [zero_tran], render=False)
                    viewer.render()
                
                idx += 1
                
                print('\r', clock.get_fps(), end='')
                
                if keyboard.is_pressed('q'):
                    break
            except Exception as e:
                print(f"Error occurred: {e}")
                print(traceback.format_exc())  # 打印完整的异常追踪信息
                os._exit(0)
            except KeyboardInterrupt:
                print("Exiting...")
                os._exit(0)
    
    accs = torch.stack(accs)
    oris = torch.stack(oris)
    # poses = torch.stack(poses)
    
    print(f"accs: {accs.shape}, oris: {oris.shape}")
    print('Frames: %d' % accs.shape[0])
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filename = f"{args.name}_{timestamp}.pt"
    
    torch.save({'acc': accs,       # [N,  5,  3]
                'ori': oris,       # [N,  5,  3,  3]
                }, os.path.join(paths.live_record_dir, data_filename))
    
    print('\rFinish.')
    os._exit(0)
        