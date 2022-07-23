

import os, sys, time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from social_vae import SocialVAE
from data import Dataloader
from utils import ADE_FDE, FPC, seed, get_rng_state, set_rng_state

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train_data", nargs='+', default=[])
parser.add_argument("--test_data", nargs='+', default=[])
parser.add_argument("--config", type=str, default=None)
parser.add_argument("--ckpt_dir", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--fpc", action="store_true", default=False)

if __name__ == "__main__":
    settings = parser.parse_args()
    config = __import__(settings.config, globals(), locals(), [
        "LEARNING_RATE", "EPOCHS", "BATCH_SIZE", "EPOCH_BATCHES",
        "NEIGHBOR_RADIUS", "OB_HORIZON", "PRED_HORIZON", "INCLUSIVE_GROUPS"
    ], 0)

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
            batch_first=False, ob_radius=config.NEIGHBOR_RADIUS,
            ob_horizon=config.OB_HORIZON, pred_horizon=config.PRED_HORIZON,
            device=settings.device, seed=settings.seed)
    train_data, test_data = None, None
    if settings.test_data:
        print(settings.test_data)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.test_data))]
        else:
            inclusive = None
        test_dataset = Dataloader(
            settings.test_data, **kwargs, inclusive_groups=inclusive,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        test_data = torch.utils.data.DataLoader(test_dataset, 
            collate_fn=test_dataset.collate_fn,
            batch_sampler=test_dataset.batch_sampler
        )
    if settings.train_data:
        print(settings.train_data)
        if config.INCLUSIVE_GROUPS is not None:
            inclusive = [config.INCLUSIVE_GROUPS for _ in range(len(settings.train_data))]
        else:
            inclusive = None
        train_dataset = Dataloader(
            settings.train_data, **kwargs, inclusive_groups=inclusive, 
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
    model = SocialVAE(horizon=config.PRED_HORIZON, observe_radius=config.NEIGHBOR_RADIUS)
    model.to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    start_epoch = 0
    if settings.ckpt_dir:
        ckpt = os.path.join(settings.ckpt_dir, "ckpt-last")
        ckpt_best = os.path.join(settings.ckpt_dir, "ckpt-best")
        if os.path.exists(ckpt_best):
            state_dict = torch.load(ckpt_best, map_location=settings.device)
            ade_best = state_dict["ade"]
            fde_best = state_dict["fde"]
        else:
            ade_best = 100000
            fde_best = 100000
        if not settings.train_data:
            ckpt = ckpt_best
        if os.path.exists(ckpt):
            print("Load from ckpt:", ckpt)
            state_dict = torch.load(ckpt, map_location=settings.device)
            model.load_state_dict(state_dict["model"])
            if "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
                rng_state = [r.to("cpu") if torch.is_tensor(r) else r for r in state_dict["rng_state"]]
            if "epoch" in state_dict:
                start_epoch = state_dict["epoch"]
    end_epoch = start_epoch+1 if train_data is None or start_epoch >= settings.epochs else settings.epochs


    if settings.train_data and settings.ckpt_dir:
        logger = SummaryWriter(log_dir=settings.ckpt_dir)
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
        if train_data is not None and epoch <= settings.epochs:
            print("Epoch {}/{}".format(epoch, config.EPOCHS))
            tic = time.time()
            set_rng_state(rng_state, settings.device)
            losses = {}
            model.train()
            sys.stdout.write(log_str.format(
                cur_batch=0, done="", remain="."*int(batches*progress),
                time=round(time.time()-tic), comment=""))
            for batch, item in enumerate(train_data):
                if train_dataset.device != device:
                    item = [_.to(device) for _ in item]
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
        ade, fde = -1, -1
        if test_data is not None:
            sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                0, len(test_dataset)
            ))
            tic = time.time()
            model.eval()
            ADE, FDE = [], []
            set_rng_state(init_rng_state, settings.device)
            batch = 0
            with torch.no_grad():
                for x, y, neighbor in test_data:
                    batch += x.size(1)
                    sys.stdout.write("\r\033[K Evaluating...{}/{}".format(
                        batch, len(test_dataset)
                    ))
                    
                    if settings.fpc:
                        y_ = []
                        for _ in range(5):
                            y_.append(model(x, neighbor, n_predictions=20))
                        y_ = torch.cat(y_, 0)
                        cand = []
                        for i in range(y_.size(-2)):
                            cand.append(FPC(y_[..., i, :].cpu().numpy(), n_samples=20))
                        # n_samples x PRED_HORIZON x N x 2
                        y_ = torch.stack([y_[_,:,i] for i, _ in enumerate(cand)], 2)
                    else:
                        # 20 x PRED_HORIZON x N x 2
                        y_ = model(x, neighbor, n_predictions=20)
                    ade, fde = ADE_FDE(y_, y)
                    ade = torch.min(ade, dim=0)[0]
                    fde = torch.min(fde, dim=0)[0]
                    ADE.append(ade)
                    FDE.append(fde)


            ADE = torch.cat(ADE)
            FDE = torch.cat(FDE)
            ade = ADE.mean()
            fde = FDE.mean()
            sys.stdout.write("\r\033[K ADE: {:.2f}; FDE: {:.2f} -- time: {}s".format(ade, fde, int(time.time()-tic)))
            print()

        ###############################################################################
        #####                                                                    ######
        ##### log                                                                ######
        #####                                                                    ######
        ###############################################################################
        if losses is not None and settings.ckpt_dir:
            if logger is not None:
                for k, v in losses.items():
                    logger.add_scalar("train/{}".format(k), v, epoch)
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
