from pygame.time import Clock
from articulate.utils.bullet.view_rotation_np import RotationViewer
from articulate.utils.pygame import StreamingDataViewer
from scipy.spatial.transform import Rotation as R
from argparse import ArgumentParser
from sensor_huawei.sensor import HuaweiSensor, CalibratedHuaweiSensor
from config import HuaweiDevices
from articulate.utils.noitom.PN_lab import IMUSet, CalibratedIMUSet
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
    parser.add_argument('--huawei', action='store_true', help='use huawei imu')
    parser.add_argument('--noitom', action='store_true', help='use noitom imu')
    parser.add_argument('--view_ori', action='store_true', help='view orientation data')
    parser.add_argument('--view_acc', action='store_true', help='view acceleration data')
    parser.add_argument('--view_t', action='store_true', help='view timestamp data')
    args = parser.parse_args()

    clock = Clock()

    if args.view_acc:
        sviewer = StreamingDataViewer(3, y_range=(-20, 20), window_length=200, names=['X', 'Y', 'Z']); sviewer.connect()
    if args.view_ori:
        if args.noitom:
            rviewer = RotationViewer(2, order='wxyz'); rviewer.connect()
        else:
            rviewer = RotationViewer(1, order='wxyz'); rviewer.connect()            
        
        rviewer_quan = StreamingDataViewer(3, y_range=(-90, 90), window_length=200, names=['x', 'y', 'z']); rviewer_quan.connect()
    if args.noitom:
        imu_set = CalibratedIMUSet()
        imu_set.calibrate('walking_9dof')
    if args.huawei:
        huawei_sensor = CalibratedHuaweiSensor(device_ids=HuaweiDevices.device_ids)
        huawei_sensor.calibrate("walking_6dof")

    ids = [id for id in HuaweiDevices.device_ids.values()]
    while True:
        try:
            clock.tick(30)
            r_list = []

            if args.huawei and not args.noitom:
                t, aS, aI, aM, RIS, RMB = huawei_sensor.get()

            elif args.noitom and not args.huawei:
                _, RIS, aS, wS, mS, aI, wI, mI, RMB, aM, wM, mM = imu_set.get()
            elif args.huawei and args.noitom:
                RMB = huawei_sensor.get()[-1]
                RMB_gt = imu_set.get()[-4]
            
            if args.view_ori:
                # qulitative visualization
                R_huawei = RMB[2]
                q_huawei = R.from_matrix(R_huawei).as_quat()
                # convert to wxyz
                q_huawei = np.array([q_huawei[3], q_huawei[0], q_huawei[1], q_huawei[2]])
                
                if args.noitom:
                    R_gt = RMB_gt[4]
                    q_gt = R.from_matrix(R_gt).as_quat()
                    q_gt = np.array([q_gt[3], q_gt[0], q_gt[1], q_gt[2]])
                    rviewer.update_all([q_gt, q_huawei])
                else:           
                    rviewer.update_all([q_huawei])

                # quantitative visualization
                delta_R = R_huawei.T @ R_gt
                delta_euler = R.from_matrix(delta_R).as_euler('xyz', degrees=True)
                rviewer_quan.plot(delta_euler)
                
            if args.view_acc:
                # process acceleration data
                sviewer.plot(aI[2])
                # print t shape
        except Exception as e:
            print(f"Error occurred: {e}")
            print(traceback.format_exc())  # 打印完整的异常追踪信息
            os._exit(0)
        except KeyboardInterrupt:
            print("Exiting...")
            os._exit(0)
