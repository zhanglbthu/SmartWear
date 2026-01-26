import socket
import threading
import os
import numpy as np
import torch
import articulate as art
import winsound
import cv2
import time

class DataReceiver(threading.Thread):
    def __init__(self, sock, sensor, buffer_size=1024):
        super().__init__(daemon=True)
        self.sock = sock
        self.sensor = sensor
        self.buffer_size = buffer_size
        self.running = True

    def run(self):
        buffer = ""
        while self.running:
            try:
                data, _ = self.sock.recvfrom(self.buffer_size)
                buffer += data.decode()

                packets = buffer.split('$')
                buffer = packets[-1]

                for pkt in packets[:-1]:
                    self.sensor.process_data(pkt)

            except Exception as e:
                print("Receiver error:", e)
                os._exit(0)

    def stop(self):
        self.running = False
        
class AndroidSensor:
    g = 9.8

    def __init__(self, ip="0.0.0.0", port=8989, buffer_size=256, device_ids=None):
        """
        device_ids: dict, e.g. {"phone": 0, "watch": 1}
        """
        self.device_ids = device_ids or {}
        self.ids = list(self.device_ids.values())
        self.N = len(self.ids)

        # socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # buffers
        self.raw_acc_buffer   = {i: np.zeros((buffer_size, 3)) for i in self.ids}
        self.raw_ori_buffer   = {i: np.tile(np.array([[1, 0, 0, 0]]), (buffer_size, 1))for i in self.ids}
        self.timestamp_buffer = {i: np.zeros((buffer_size, 1)) for i in self.ids}

        # receiver thread
        self.receiver = DataReceiver(self.sock, self, buffer_size=1024)
        self.receiver.start()
    
    def process_data(self, msg: str):
        """
        msg format:
        sensor_id#timestamp#orientation#acceleration#raw_gyro#raw_acc#raw_mag#...
        """
        try:
            parts = msg.split('#')
            dev_id = int(parts[0])
            timestamp = float(parts[1])

            orientation = np.array([float(x) for x in parts[2].split()])  # w x y z
            acceleration = np.array([float(x) * self.g for x in parts[3].split()])  # xyz (m/s^2)

        except Exception as e:
            print("Parse error:", e, msg)
            return

        if dev_id not in self.raw_acc_buffer:
            return

        # shift buffer
        self.raw_acc_buffer[dev_id] = np.concatenate([self.raw_acc_buffer[dev_id][1:], acceleration[None]], axis=0)

        self.raw_ori_buffer[dev_id] = np.concatenate([self.raw_ori_buffer[dev_id][1:], orientation[None]],  axis=0)

        self.timestamp_buffer[dev_id] = np.concatenate([self.timestamp_buffer[dev_id][1:], [[timestamp]]],  axis=0)

    def get_latest_data(self, device_ids):
        acc = [self.raw_acc_buffer[i][-1] for i in device_ids]
        ori = [self.raw_ori_buffer[i][-1] for i in device_ids]
        t   = [self.timestamp_buffer[i][-1] for i in device_ids]

        return np.array(acc), np.array(ori), np.array(t)

    def get(self, device_ids=None):
        if device_ids is None:
            device_ids = self.ids

        aS, quat, timestamp = self.get_latest_data(device_ids)

        # quat: wxyz → rotation matrix
        RIS = art.math.quaternion_to_rotation_matrix(
            torch.tensor(quat).float()
        )  # [N, 3, 3]

        aS = torch.tensor(aS, dtype=torch.float32)
        
        # change: convert as from left-hand to right-hand
        aS = aS * torch.tensor([-1, -1, -1]).float()
        
        aI = RIS.matmul(aS.unsqueeze(-1)).squeeze(-1)

        timestamp = torch.tensor(timestamp).float()

        return timestamp, aS, aI, RIS

class CalibratedAndroidSensor(AndroidSensor):
    _RMB_Npose = torch.tensor([[[0, 1, 0], [-1, 0, 0], [0, 0, 1]],             # left wrist
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]],             # right wrist
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],              # left thigh
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],              # right thigh
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],]).float()    # head
                            
    
    # _RMB_Npose = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]],            # right thigh
    #                            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],           # left wrist
    #                            [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float()   # head
    
    def __init__(self, device_ids, buffer_size=1024):
        super().__init__(device_ids=device_ids, buffer_size=buffer_size)
        self.ids = [id for id in device_ids.values()]
        self.N = len(self.ids)
        self.RMI = torch.eye(3).repeat(self.N, 1, 1)
        self.RSB = torch.eye(3).repeat(self.N, 1, 1)
        
        # select Npose using device ids
        self.Npose = self._RMB_Npose[self.ids]
        
        # print self.ids
        print("Device IDs:", self.device_ids)

    def get_cali_matrices(self):
        return self.RMI.clone(), self.RSB.clone()
    
    def get(self):
        t, aS, aI, RIS = super().get(self.ids) # aS: [N, 3], RIS: [N, 3, 3] 

        RMB = self.RMI.matmul(RIS).matmul(self.RSB)
        aM = self.RMI.matmul(aI.unsqueeze(-1)).squeeze(-1)
        
        return t, aS, aI, aM, RIS, RMB

    @staticmethod
    def _normalize_tensor(x: torch.Tensor, dim=-1):
        norm = x.norm(dim=dim, keepdim=True)
        normalized_x = x / norm
        return normalized_x

    @staticmethod
    def _mean_rotation(R0, R1):
        R_avg = (R0 + R1) / 2
        U, S, V = torch.svd(R_avg)
        R = torch.matmul(U, V.transpose(-2, -1)).view(-1, 3, 3)
        m = R.det() < 0
        R[m] = U[m].matmul(torch.diag(torch.tensor([1, 1, -1.], device=R.device))).bmm(V[m].transpose(1, 2))
        return R
    @staticmethod
    def _rotation_matrix_to_axis_angle(r):
        result = [cv2.Rodrigues(_)[0] for _ in r.view(-1, 3, 3).numpy()]
        result = torch.from_numpy(np.stack(result)).float().squeeze(-1)
        return result
    @staticmethod
    def _input(s, skip_input):
        if skip_input:
            print(s)
        else:
            return input(s)

    def _walking_calibration(self, RMI_method, skip_input=False):
        self._input('Stand in N pose for 3 seconds, then step forward and stop in the N-pose again.', skip_input)
        time.sleep(3)
        RIS_N0 = self.get()[4]

        winsound.Beep(440, 600)
        print('Step forward now.')
        begin_t = last_t = self.get()[0][0]
        p, v = torch.zeros(self.N, 3), torch.zeros(self.N, 3)
        Cov_pv = Cov_vv = 0
        
        while last_t - begin_t < 3:
            t, _, aI, _, _, _ = self.get()
            t = t[0][0]
            if t != last_t:
                dt = 0.01
                last_t = t
                p += dt * v + 0.5 * dt * dt * aI
                v += dt * aI
                Cov_pv += dt * Cov_vv
                Cov_vv += 1
        p_filtered = p - Cov_pv / Cov_vv * v
        RIS_N1 = self.get()[4]

        if RMI_method == '9dof':
            zI = p_filtered.mean(dim=0).repeat(self.N, 1)
        elif RMI_method == '6dof':
            zI = p_filtered
        yI = torch.tensor([0, 0, 1.]).expand(self.N, 3)
        xI = self._normalize_tensor(yI.cross(zI, dim=-1))
        zI = self._normalize_tensor(xI.cross(yI, dim=-1))
        RMI = torch.stack([xI, yI, zI], dim=-2)
        
        RSB0 = RMI.matmul(RIS_N0).transpose(1, 2).matmul(self.Npose)
        RSB1 = RMI.matmul(RIS_N1).transpose(1, 2).matmul(self.Npose)
        
        RSB = self._mean_rotation(RSB0, RSB1)

        err_vertical = p_filtered[:, -1].abs()
        err_RSB = self._rotation_matrix_to_axis_angle(RSB0.bmm(RSB1.transpose(1, 2))).norm(dim=-1) * (180 / np.pi)
        # change err_vertical from 0.1 to 0.4
        if all(err_vertical < 0.4) and all(err_RSB < 20) or skip_input:
            c = 'n'
            print('Calibration succeed: vertical error %s m, RSB error %s deg' % (err_vertical, err_RSB))
        else:
            c = input('Calibration fail: vertical error %s m, RSB error %s deg. Try again? [y]/n' % (err_vertical, err_RSB))
        if c != 'n':
            self._walking_calibration(RMI_method, skip_input)
        else:
            self.RMI = RMI
            self.RSB = RSB

    def calibrate(self, method, skip_input=False):
        r"""
        Calibrate the IMU set.

        :param method: Calibration method. Select from:
            - 'tpose_9dof':     Full T-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in T-pose.
            - 'tpose_6dof':     Full T-pose calibration for 6-dof IMU. Two steps: align all imus and stand in T-pose.
            - 'tpose_onlyRSB':  T-pose calibration only for sensor-to-bone offset. One step: stand in T-pose.
            - 'npose_9dof':     Full N-pose calibration for 9-dof IMU. Two steps: align imu 1 and stand in N-pose.
            - 'npose_6dof':     Full N-pose calibration for 6-dof IMU. Two steps: align all imus and stand in N-pose.
            - 'npose_onlyRSB':  N-pose calibration only for sensor-to-bone offset. One step: stand in N-pose.
            - 'npose_tpose':    Change-pose calibration for 9-dof IMU. One step: stand in N pose and then change to T-pose.
            - 'walking_9dof':   Walking calibration for 9-dof IMU. One step: stand in N pose, step forward and stop in N pose.
            - 'walking_6dof':   Walking calibration for 6-dof IMU. One step: stand in N pose, step forward and stop in N pose.
        :param skip_input: If true, skip user input and do calibration directly without error check.
        """
        if method == 'tpose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='tpose', skip_input=skip_input)
        elif method == 'tpose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='tpose', skip_input=skip_input)
        elif method == 'npose_9dof':
            self._fixpose_calibration(RMI_method='9dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_6dof':
            self._fixpose_calibration(RMI_method='6dof', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_onlyRSB':
            self._fixpose_calibration(RMI_method='skip', RSB_method='npose', skip_input=skip_input)
        elif method == 'npose_tpose':
            self._changepose_calibration(skip_input=skip_input)
        elif method == 'walking_9dof':
            self._walking_calibration(RMI_method='9dof', skip_input=skip_input)
        elif method == 'walking_6dof':
            self._walking_calibration(RMI_method='6dof', skip_input=skip_input)
