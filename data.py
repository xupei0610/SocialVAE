from typing import Optional, Sequence, List

import os, sys
import pickle

import torch
import numpy as np


class Dataloader(torch.utils.data.Dataset):
    def __init__(self, 
        files: List[str], ob_horizon: int, pred_horizon: int,
        frameskip: int=1, exclusive_agent_ids: Optional[Sequence]=None,
        batch_first: bool=False, seed: Optional[int]=None,
        device: Optional[torch.device]=None,
        flip: bool=False, rotate: bool=False, scale: bool=False
    ):
        super().__init__()
        self.ob_horizon = ob_horizon
        self.batch_first = batch_first
        self.flip = flip
        self.rotate = rotate
        self.scale = scale
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu") 
        else:
            self.device = device
        if seed:
            self.rng = np.random.RandomState()
            self.rng.seed(seed)
        else:
            self.rng = np.random
        self.data = []

        frameskip = int(frameskip) if frameskip and int(frameskip) > 1 else 1

        if exclusive_agent_ids is None:
            exclusive_agent_ids = [[] for _ in range(len(files))]
        assert(len(exclusive_agent_ids) == len(files))

        files = [(f, exclusive) for f, exclusive in zip(files, exclusive_agent_ids)]
        for f, exclusive in files:
            if os.path.isdir(f):
                for ff in os.listdir(f):
                     if not ff.startswith(".") and not ff.startswith("_"):
                        files.append((os.path.join(f, ff), exclusive))
        files = [(f, exc) for f, exc in files if os.path.exists(f) and not os.path.isdir(f)]
        files = sorted(files, key=lambda _: _[0])
        
        self.files = []
        self.data_len = {}
        total_data = 0
        sys.stdout.write("\r\033[K Loading...{}/{}".format(
            0, len(files)
        ))
        for fid, (f, exclusive) in enumerate(files):
            sys.stdout.write("\r\033[K Loading...{}/{}".format(
                fid+1, len(files)
            ))
            self.files.append(f)
            with open(f, "r") as record:
                data = self.load(record)
            if len(data) > (ob_horizon+pred_horizon-1)*frameskip:
                data = self.extend(data, frameskip)
                self.load_traj(data, ob_horizon, pred_horizon, frameskip, exclusive)
            self.data_len[f] = len(self.data) - total_data
            total_data = len(self.data)
        print("\n   {} trajectories loaded.".format(total_data))
        
    def collate_fn(self, batch):
        X, Y, NEIGHBOR = [], [], []
        for traj, neighbor in batch:
            traj_shape = traj.shape
            neighbor_shape = neighbor.shape
            traj = np.reshape(traj, (-1, 2))
            neighbor = np.reshape(neighbor, (-1, 2))
            if self.flip:
                if self.rng.randint(2):
                    traj[..., 1] *= -1
                    neighbor[..., 1] *= -1
                if self.rng.randint(2):
                    traj[..., 0] *= -1
                    neighbor[..., 0] *= -1
            if self.rotate:
                d = self.rng.random() * (np.pi+np.pi) 
                s, c = np.sin(d), np.cos(d)
                R = np.asarray([
                    [c, -s],
                    [s,  c]
                ])
                traj = (R @ np.expand_dims(traj, -1))[..., 0]
                neighbor = (R @ np.expand_dims(neighbor, -1))[..., 0]
            if self.scale:
                s = self.rng.randn()*0.05 + 1 # N(1, 0.05)
                traj = s * traj
                neighbor = s * neighbor
            traj = np.reshape(traj, traj_shape)
            neighbor = np.reshape(neighbor, neighbor_shape)
            X.append(traj[:self.ob_horizon])
            Y.append(traj[self.ob_horizon:,...,:2])
            NEIGHBOR.append(neighbor)
        
        n_neighbors = [n.shape[1] for n in NEIGHBOR]
        max_neighbors = max(n_neighbors) 
        if max_neighbors != min(n_neighbors):
            NEIGHBOR = [
                np.pad(neighbor, ((0, 0), (0, max_neighbors-n), (0, 0)), 
                "constant", constant_values=1e9)
                for neighbor, n in zip(NEIGHBOR, n_neighbors)
            ]
        stack_dim = 0 if self.batch_first else 1
        X = np.stack(X, stack_dim)
        Y = np.stack(Y, stack_dim)
        NEIGHBOR = np.stack(NEIGHBOR, stack_dim)

        X = torch.tensor(X, device=self.device, dtype=torch.float32)
        Y = torch.tensor(Y, device=self.device, dtype=torch.float32)
        NEIGHBOR = torch.tensor(NEIGHBOR, device=self.device, dtype=torch.float32)
        return X, Y, NEIGHBOR

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def load_traj(self, data, ob_horizon, pred_horizon, frameskip, exclusive):
        time = np.sort(list(data.keys()))
        horizon = (ob_horizon+pred_horizon-1)*frameskip

        tid0 = 0
        while tid0 < len(time)-horizon:
            tid1 = tid0+horizon
            t0 = time[tid0]
            dt = time[tid0+1] - t0

            idx = [_ for _ in data[t0].keys() if _ not in exclusive]
            idx_all = list(data[t0].keys())
            if idx:
                for tid in range(tid0+frameskip, tid1+1, frameskip):
                    t = time[tid]
                    if t - time[tid-1] != dt: # ignore non-continuous frames
                        tid0 = tid-1
                        idx = []
                        break
                    idx_cur = [_ for _ in data[t].keys() if _ not in exclusive]
                    if not idx_cur: # ignore empty frames
                        tid0 = tid
                        idx = []
                        break
                    idx = np.intersect1d(idx, idx_cur)
                    if len(idx) == 0: break
                    idx_all.extend(idx_cur)
            if len(idx):
                idx_all = np.concatenate((idx, np.setdiff1d(idx_all, idx)))
                for i in idx:
                    data_dim = len(data[time[tid0]][i])
                    if len(idx_all) == 1:
                        agent = [data[time[tid]][i] for tid in range(tid0, tid1+1, frameskip)]
                        neighbor = [[[1e9]*data_dim] for _ in range(len(agent))]
                    else:
                        agent, neighbor = [], []
                        for tid in range(tid0, tid1+1, frameskip):
                            t = time[tid]
                            agent.append(data[t][i])
                            neighbor.append([
                                data[t][j] if j in data[t] else [1e9]*data_dim
                                for j in idx_all if i != j
                            ])
                    self.data.append((np.float32(agent), np.float32(neighbor)))
            tid0 += 1

                
    def extend(self, data, frameskip):
        time = np.sort(list(data.keys()))
        dts = np.unique(time[1:] - time[:-1])
        dt = dts.min()
        if np.any(dts % dt != 0):
            raise ValueError("Inconsistent frame interval:", dts)
        i = 0
        while i < len(time)-1:
            if time[i+1] - time[i] != dt:
                np.insert(time, i+1, time[i]+dt)
            i += 1
        # ignore those only appearing at one frame
        for tid, t in enumerate(time):
            removed = []
            if t not in data: data[t] = {}
            for idx in data[t].keys():
                t0 = time[tid-frameskip] if tid >= frameskip else None
                t1 = time[tid+frameskip] if tid+frameskip < len(time) else None
                if (t0 is None or idx not in data[t0]) and (t1 is None or idx not in data[t1]):
                    removed.append(idx)
            for idx in removed:
                data[t].pop(idx)
        # extend v
        for tid in range(len(time)-frameskip):
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                x0 = data[t0][i][0]
                y0 = data[t0][i][1]
                x1 = data[t1][i][0]
                y1 = data[t1][i][1]
                vx, vy = x1-x0, y1-y0
                data[t1][i].insert(2, vx)
                data[t1][i].insert(3, vy)
                if tid < frameskip or i not in data[time[tid-1]]:
                    data[t0][i].insert(2, vx)
                    data[t0][i].insert(3, vy)
        # extend a
        for tid in range(len(time)-frameskip):
            t_1 = None if tid < frameskip else time[tid-frameskip]
            t0 = time[tid]
            t1 = time[tid+frameskip]
            if t1 not in data or t0 not in data: continue
            for i, item in data[t1].items():
                if i not in data[t0]: continue
                vx0 = data[t0][i][2]
                vy0 = data[t0][i][3]
                vx1 = data[t1][i][2]
                vy1 = data[t1][i][3]
                ax, ay = vx1-vx0, vy1-vy0
                data[t1][i].insert(4, ax)
                data[t1][i].insert(5, ay)
                if t_1 is None or i not in data[t_1]:
                    # first appearing frame, pick value from the next frame
                    data[t0][i].insert(4, ax)
                    data[t0][i].insert(5, ay)
        return data

    def load(self, file):
        data = {}
        for row in file.readlines():
            item = row.split()
            if not item: continue
            t = int(float(item[0]))
            idx = int(float(item[1]))
            x = float(item[2])
            y = float(item[3])
            if t not in data:
                data[t] = {}
            data[t][idx] = [x, y]
        return data
