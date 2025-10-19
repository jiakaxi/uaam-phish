import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from src.utils.seed import set_global_seed
from src.systems.url_only_module import UrlOnlySystem
from src.datamodules.url_datamodule import UrlDataModule
import torch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, choices=[None, "local", "server"])
    args = ap.parse_args()

    cfg = OmegaConf.load("configs/default.yaml")
    if args.profile:
        prof = OmegaConf.load(f"configs/profiles/{args.profile}.yaml")
        cfg = OmegaConf.merge(cfg, prof)

    set_global_seed(cfg.train.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    dm = UrlDataModule(cfg)
    sys = UrlOnlySystem(cfg)
    # 使用配置文件中的回调参数
    monitor = cfg.eval.get("monitor", "val/loss")
    patience = cfg.eval.get("patience", 3)
    
    # 根据监控指标确定模式
    mode = "max" if "f1" in monitor or "auroc" in monitor else "min"
    
    callbacks = [
        EarlyStopping(monitor=monitor, mode=mode, patience=patience),
        ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1,
                       filename=f"best-{{epoch}}-{{{monitor.replace('/', '_')}:.3f}}")
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        strategy=cfg.hardware.get("strategy", "auto"),
        log_every_n_steps=cfg.train.log_every,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )
    print(f"[CFG] model={cfg.model.pretrained_name}, max_len={cfg.data.max_length}, "
          f"epochs={cfg.train.epochs}, bs={cfg.train.bs}, sample_frac={cfg.data.sample_fraction}, "
          f"accel={cfg.hardware.accelerator}")
    print(f"[CALLBACKS] monitor={monitor}, mode={mode}, patience={patience}")
    
    trainer.fit(sys, dm)
    trainer.test(sys, datamodule=dm)
