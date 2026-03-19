from pygame.time import Clock
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
from sensor_android.sensor import AndroidSensor, CalibratedAndroidSensor
from sensor_apple.sensor import AppleSensor, CalibratedAppleSensor
from sensor_huawei.sensor import HuaweiSensor, CalibratedHuaweiSensor
from config import AndroidDevices, HuaweiDevices, AppleDevices
import articulate as art
import torch
import os
import traceback
import numpy as np

np.set_printoptions(precision=10, suppress=True)
torch.set_printoptions(sci_mode=False)
# 主程序代码
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--android', action='store_true', help='use android imu')
    parser.add_argument('--apple', action='store_true', help='use apple imu')
    parser.add_argument('--huawei', action='store_true', help='use huawei imu')
    parser.add_argument('--idx', type=int, default=0, help='device index')
    parser.add_argument('--view_ori', action='store_true', help='view orientation data')
    parser.add_argument('--view_acc', action='store_true', help='view acceleration data')
    parser.add_argument('--view_sync', action='store_true', help='view sync status')
    args = parser.parse_args()

    clock = Clock()        
    
    if args.android:
        sensor = CalibratedAndroidSensor(device_ids=AndroidDevices.device_ids)
        sensor.calibrate("walking_6dof")
        ids = [id for id in AndroidDevices.device_ids.values()]
    elif args.huawei:
        sensor = CalibratedHuaweiSensor(device_ids=HuaweiDevices.device_ids)
        sensor.calibrate("walking_6dof")
        ids = [id for id in HuaweiDevices.device_ids.values()]
    elif args.apple:
        sensor = CalibratedAppleSensor(AppleDevices.udp_ports, device_ids=AppleDevices.device_ids)
        sensor.calibrate("walking_6dof")
        ids = [id for id in AppleDevices.device_ids.values()]
    else:
        print("No valid sensor selected.")
        os._exit(0)

    if args.view_acc:
        sviewer = StreamingDataViewer(3, y_range=(-20, 20), window_length=200, names=['X', 'Y', 'Z']); sviewer.connect()
    if args.view_sync:
        sviewer = StreamingDataViewer(len(ids), y_range=(-20, 20), window_length=200, names=[f'Device {id}' for id in ids]); sviewer.connect()
    if args.view_ori:
        rviewer = RotationViewer(1, order='wxyz'); rviewer.connect()    

    id = args.idx
    acc_list = []
    
    while True:
        try:
            clock.tick(30)
            print(clock.get_fps(), end='')

            t, aS, aI, aM, RIS, RMB = sensor.get()
            
            if args.view_ori:
                r_list = []

                RMB_i = RMB[id]
                q_i = R.from_matrix(RMB_i).as_quat()

                rviewer.update_all([q_i])

            if args.view_acc:
                # print acc
                acc_list.append(aI[id].cpu().numpy())
                # if len(acc_list) > 200:
                #     # calculate mean and std
                #     acc_array = np.array(acc_list)
                #     mean = np.mean(acc_array, axis=0)
                #     std = np.std(acc_array, axis=0)
                #     print(f"Mean: {mean}, Std: {std}")
                #     os._exit(0)
                sviewer.plot(aI[id])
            if args.view_sync:
                # 可视化每个设备的aI的模长，检查同步状态
                sync_values = [torch.norm(aI[i]).item() for i in range(len(ids))]
                sviewer.plot(sync_values)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            os._exit(0)
        except KeyboardInterrupt:
            print("Exiting...")
            os._exit(0)
