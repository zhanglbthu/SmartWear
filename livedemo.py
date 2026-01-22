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
from apple_sensor.sensor import AppleSensor, CalibratedAppleSensor
from huawei_sensor.sensor import HuaweiSensor, CalibratedHuaweiSensor
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
        
    combo = [0, 3, 4]

    if args.noitom:
        # add ground truth readings
        imu_set = CalibratedIMUSet()
        imu_set.calibrate('walking_9dof')
    
    if args.apple:
        apple_sensor = CalibratedAppleSensor(AppleDevices.udp_ports, AppleDevices.device_ids)
        apple_sensor.calibrate("walking_6dof")
        RMI, RSB = apple_sensor.get_cali_matrices()
        RMI = RMI.permute(1, 0, 2).to(device)
        RSB = RSB.permute(1, 0, 2).to(device)

    if args.huawei:
        huawei_sensor = CalibratedHuaweiSensor(HuaweiDevices.device_ids)
        huawei_sensor.calibrate("walking_6dof")

    accs, oris, poses, trans = [], [], [], []
    accs_gt, oris_gt = [], []
    
    idx = 0
    # sviewer = StreamingDataViewer(3, y_range=(-90, 90), window_length=200, names=['Y', 'Z', 'X']); sviewer.connect()
    # rviewer = RotationViewer(2, order='wxyz'); rviewer.connect()
    with torch.no_grad(), MotionViewer(1, overlap=False, names=['huawei_mocap']) as viewer:
        while True:
            try:
                clock.tick(30)
                # viewer.clear_all(render=False)
                ori = torch.zeros(5, 3, 3).to(device)
                a = torch.zeros(5, 3).to(device)
                if args.noitom:
                    # gt readings
                    _, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM = imu_set.get()
                    
                    ori[combo] = RMB[combo].to(device)
                    a[combo] = aM[combo].to(device)
                    
                    oris_gt.append(ori.clone())
                    accs_gt.append(a.clone())
                    
                if args.apple:
                    # device readings
                    t, aS, aI, aM, RIS, RMB = apple_sensor.get()
                    ori[combo] = RMB[[1, 0, 2]].to(device)
                    a[combo] = aM[[1, 0, 2]].to(device)

                    oris.append(ori.clone())
                    accs.append(a.clone())

                if args.huawei:
                    t, aS, aI, aM, RIS, RMB = huawei_sensor.get()
                    ori[combo] = RMB[[1, 0, 2]].to(device)
                    a[combo] = aM[[1, 0, 2]].to(device)

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
            
                # # 去除ori_gt和ori的最后一个数据
                # cur_ori_gt = oris_gt[-1]
                # cur_ori = oris[-1]
                
                # R_err = cur_ori.transpose(-2, -1).matmul(cur_ori_gt)
                # euler_err = art.math.rotation_matrix_to_euler_angle(R_err, seq='YZX') * 180 / np.pi
                # euler_err = euler_err.view(5, 3)

                # device_idx = args.device_idx
                # sviewer.plot(euler_err[device_idx].cpu().numpy())
                
                # ori_mat = art.math.rotation_matrix_to_axis_angle(cur_ori)
                # ori_gt_mat = art.math.rotation_matrix_to_axis_angle(cur_ori_gt)
                # ori_q = art.math.axis_angle_to_quaternion(ori_mat)
                # ori_gt_q = art.math.axis_angle_to_quaternion(ori_gt_mat)
                # rviewer.update_all([ori_gt_q[device_idx].cpu().numpy(), ori_q[device_idx].cpu().numpy()])
                
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
    accs_gt = torch.stack(accs_gt)
    oris_gt = torch.stack(oris_gt)
    # poses = torch.stack(poses)
    
    print(f"accs: {accs.shape}, oris: {oris.shape}, accs_gt: {accs_gt.shape}, oris_gt: {oris_gt.shape}")
    print(f"RMI: {RMI.shape}, RSB: {RSB.shape}")
    print('Frames: %d' % accs.shape[0])
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    data_filename = f"{args.name}_{timestamp}.pt"
    
    torch.save({'acc': accs,       # [N,  5,  3]
                'ori': oris,       # [N,  5,  3,  3]
                'acc_gt': accs_gt, # [N,  5, 3]
                'ori_gt': oris_gt, # [N,  5,  3,  3]
                'RMI': RMI,        # [3, 3, 3]
                'RSB': RSB         # [3, 3, 3]
                }, os.path.join(paths.live_record_dir, data_filename))
    
    print('\rFinish.')
    os._exit(0)
        