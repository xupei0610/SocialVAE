import os, sys, time
import importlib
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from social_vae import SocialVAE
from data import Dataloader
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", nargs='+', default=[])
parser.add_argument("--test", nargs='+', default=[])
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--no-fpc", action="store_true", default=False)
parser.add_argument("--fpc-finetune", action="store_true", default=False)

if __name__ == "__main__":
    settings = parser.parse_args()
    spec = importlib.util.spec_from_file_location("config", settings.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if settings.device is None:
        settings.device = "cuda" if torch.cuda.is_available() else "cpu"
    settings.device = torch.device(settings.device)
    
    seed(settings.seed)
    init_rng_state = get_rng_state(settings.device)
    rng_state = init_rng_state

    ###############################################################################
    #####                                                                    ######
    ##### prepare datasets                                                   ######
    #####                                                                    ######
    ###############################################################################
    kwargs = dict(
            batch_first=False,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None
    if settings.test:
        print(settings.test)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            settings.test, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )
        def test(model, fpc=1):
            sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                0, len(test_dataset)
            ))
            tic = time.time()
            model.eval()
            ADE, FDE = [], []
            set_rng_state(init_rng_state, settings.device)
            batch = 0
            fpc = int(fpc) if fpc else 1
            fpc_config = "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC"
            with torch.no_grad():
                for x, y, neighbor in test_data:
                    batch += x.size(1)
                    sys.stdout.write("\r\033[K Evaluating...{}/{} ({}) -- time: {}s".format(
                        batch, len(test_dataset), fpc_config, int(time.time()-tic)
                    ))
                    
                    if config.PRED_SAMPLES > 0 and fpc > 1:
                        # disable fpc testing during training
                        y_ = []
                        for _ in range(fpc):
                            y_.append(model(x, neighbor, n_predictions=config.PRED_SAMPLES))
                        y_ = torch.cat(y_, 0)
                        cand = []
                        for i in range(y_.size(-2)):
                            cand.append(FPC(y_[..., i, :].cpu().numpy(), n_samples=config.PRED_SAMPLES))
                        # n_samples x PRED_HORIZON x N x 2
                        y_ = torch.stack([y_[_,:,i] for i, _ in enumerate(cand)], 2)
                    else:
                        # n_samples x PRED_HORIZON x N x 2
                        y_ = model(x, neighbor, n_predictions=config.PRED_SAMPLES)
                    ade, fde = ADE_FDE(y_, y)
                    if config.PRED_SAMPLES > 0:
                        ade = torch.min(ade, dim=0)[0]
                        fde = torch.min(fde, dim=0)[0]
                    ADE.append(ade)
                    FDE.append(fde)
            ADE = torch.cat(ADE)
            FDE = torch.cat(FDE)
            if torch.is_tensor(config.WORLD_SCALE) or config.WORLD_SCALE != 1:
                if not torch.is_tensor(config.WORLD_SCALE):
                    config.WORLD_SCALE = torch.as_tensor(config.WORLD_SCALE, device=ADE.device, dtype=ADE.dtype)
                ADE *= config.WORLD_SCALE
                FDE *= config.WORLD_SCALE
            ade = ADE.mean()
            fde = FDE.mean()
            sys.stdout.write("\r\033[K ADE: {:.4f}; FDE: {:.4f} ({}) -- time: {}s".format(
                ade, fde, fpc_config, 
                int(time.time()-tic))
            )
            print()
            return ade, fde

    if settings.train:
        print(settings.train)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train))]
        else:
            inclusive = None
        train_dataset = Dataloader(
            settings.train, **kwargs, inclusive_groups=inclusive, 
            flip=True, rotate=True, scale=True,
            batch_size=config.BATCH_SIZE, shuffle=True, batches_per_epoch=config.EPOCH_BATCHES
        )
        train_data = torch.utils.data.DataLoader(train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=train_dataset.batch_sampler
        )
        batches = train_dataset.batches_per_epoch

    ###############################################################################
    #####                                                                    ######
    ##### load model                                                         ######
    #####                                                                    ######
    ###############################################################################
    model = SocialVAE(horizon=config.PRED_HORIZON, ob_radius=config.OB_RADIUS, hidden_dim=config.RNN_HIDDEN_DIM)
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
    if settings.ckpt:
        ckpt = os.path.join(settings.ckpt, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
            fpc_best = state_dict["fpc"] if "fpc" in state_dict else 1
        else:
            ade_best = 100000
            fde_best = 100000
            fpc_best = 1
        if train_data is None: # testing mode
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            start_epoch = state_dict["epoch"]
    end_epoch = start_epoch+1 if train_data is None or start_epoch >= config.EPOCHS else config.EPOCHS

    if settings.train and settings.ckpt:
        logger = SummaryWriter(log_dir=settings.ckpt)
    else:
        logger = None

    if train_data is not None:
        log_str = "\r\033[K {cur_batch:>"+str(len(str(batches)))+"}/"+str(batches)+" [{done}{remain}] -- time: {time}s - {comment}"    
        progress = 20/batches if batches > 20 else 1
        optimizer.zero_grad()

    for epoch in range(start_epoch+1, end_epoch+1):
        ###############################################################################
        #####                                                                    ######
        ##### train                                                              ######
        #####                                                                    ######
        ###############################################################################
        losses = None
        if train_data is not None and epoch <= config.EPOCHS:
            print("Epoch {}/{}".format(epoch, config.EPOCHS))
            tic = time.time()
            set_rng_state(rng_state, settings.device)
            losses = {}
            model.train()
            sys.stdout.write(log_str.format(
                cur_batch=0, done="", remain="."*int(batches*progress),
                time=round(time.time()-tic), comment=""))
            for batch, item in enumerate(train_data):
                res = model(*item)
                loss = model.loss(*res)
                loss["loss"].backward()
                optimizer.step()
                optimizer.zero_grad()
                for k, v in loss.items():
                    if k not in losses: 
                        losses[k] = v.item()
                    else:
                        losses[k] = (losses[k]*batch+v.item())/(batch+1)
                sys.stdout.write(log_str.format(
                    cur_batch=batch+1, done="="*int((batch+1)*progress),
                    remain="."*(int(batches*progress)-int((batch+1)*progress)),
                    time=round(time.time()-tic),
                    comment=" - ".join(["{}: {:.4f}".format(k, v) for k, v in losses.items()])
                ))
            rng_state = get_rng_state(settings.device)
            print()

        ###############################################################################
        #####                                                                    ######
        ##### test                                                               ######
        #####                                                                    ######
        ###############################################################################
        ade, fde = 10000, 10000
        perform_test = (train_data is None or epoch > config.TEST_SINCE) and test_data is not None
        if perform_test:
            if not settings.no_fpc and not settings.fpc_finetune and losses is None and fpc_best > 1:
                fpc = fpc_best
            else:
                fpc = 1
            ade, fde = test(model, fpc)

        ###############################################################################
        #####                                                                    ######
        ##### log                                                                ######
        #####                                                                    ######
        ###############################################################################
        if losses is not None and settings.ckpt:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
                if perform_test:
                    logger.add_scalar("eval/ADE", ade, epoch)
                    logger.add_scalar("eval/FDE", fde, epoch)
            state = dict(
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                ade=ade, fde=fde, epoch=epoch, rng_state=rng_state
            )
            torch.save(state, ckpt)
            if ade < ade_best:
                ade_best = ade
                fde_best = fde
                state = dict(
                    model=state["model"],
                    ade=ade, fde=fde, epoch=epoch
                )
                torch.save(state, ckpt_best)

    if settings.fpc_finetune or losses is not None:
        # FPC finetune if it is specified or after training
        precision = 2
        trunc = lambda v: np.trunc(v*10**precision)/10**precision
        if settings.ckpt:
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            model.load_state_dict(state_dict["model"])
            ade_ = [trunc(state_dict["ade"].item())]
            fde_ = [trunc(state_dict["fde"].item())]
            fpc_ = [1]
        else:
            ade_, fde_, fpc_ = [], [], []
        for fpc in config.FPC_SEARCH_RANGE:
            ade, fde = test(model, fpc)
            ade_.append(trunc(ade.item()))
            fde_.append(trunc(fde.item()))
            fpc_.append(fpc)
        i = np.argmin(np.add(ade_, fde_))
        ade, fde, fpc = ade_[i], fde_[i], fpc_[i]
        if settings.ckpt:
            state_dict["ade_fpc"] = ade
            state_dict["fde_fpc"] = fde
            state_dict["fpc"] = fpc
            torch.save(state_dict, ckpt_best)
        print(" ADE: {:.2f}; FDE: {:.2f} ({})".format(
            ade, fde, "FPC: {}".format(fpc) if fpc > 1 else "w/o FPC", 
        ))
