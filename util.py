import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        """
        patience：连续多少轮不提升就停
        min_delta：loss至少下降这么多才判定为改善
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"⏱ EarlyStop counter: {self.counter}/{self.patience} (best_val={self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True

# ====================== 【2】Dataset ======================
class KlineSeqDataset(Dataset):
    def __init__(self, stacked_tokens: np.ndarray, close_ref: np.ndarray,seq_len: int):
        self.data = torch.from_numpy(stacked_tokens).long()
        self.close_ref = torch.from_numpy(close_ref).float()
        self.seq_len = seq_len
        assert len(self.data) == len(self.close_ref)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx+1 : idx + self.seq_len + 1]
        # 窗口最后一根收盘价，作为推理锚点
        last_close = self.close_ref[idx + self.seq_len - 1]
        return x, y,last_close

def get_dataloader(seq_len,batch_size,SPLIT_RATIO_1,SPLIT_RATIO_2,data_npy,close_npy):    
    # 加载预处理好的token，stacked_tokens: [N,4] int64 add_offset完成
    
    df_np = np.load(data_npy)
    close_np = np.load(close_npy)       
    
    assert len(df_np)==len(close_np)

    n_total = len(df_np)
    n_train = int(n_total * SPLIT_RATIO_1)
    n_val   = int(n_total * (SPLIT_RATIO_1+SPLIT_RATIO_2))
    train_arr = df_np[:n_train]
    val_arr = df_np[n_train:n_val]
    test_arr = df_np[n_val:]

    train_close_arr = close_np[:n_train]
    val_close_arr = close_np[n_train:n_val]
    test_close_arr = close_np[n_val:]

    print(f"total kline: {n_total}, train: {len(train_arr)}, val: {len(val_arr)} test: {len(test_arr)}")

    # 构建数据集、loader
    train_ds = KlineSeqDataset(train_arr, train_close_arr,seq_len=seq_len)
    val_ds = KlineSeqDataset(val_arr,val_close_arr, seq_len=seq_len)
    test_ds = KlineSeqDataset(test_arr,test_close_arr, seq_len=seq_len)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    return train_loader,val_loader,test_loader

def plot_loss_curve(train_hist, val_hist, save_path="loss_curve.png"):
    """
    双窗口子图：
    上子图：train loss
    下子图：val loss（自动跳过nan）
    """
    

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    x = np.arange(len(train_hist))

    # 上图：训练loss
    ax1.plot(x, train_hist, color="#1f77b4", linewidth=1.5, label="Train Loss")
    ax1.set_title("Train Loss")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 下图：验证loss，过滤nan点
    val_arr = np.array(val_hist)
    mask = ~np.isnan(val_arr)
    ax2.plot(x[mask], val_arr[mask], color="#ff7f0e", linewidth=1.5, label="Val Loss")
    ax2.set_title("Val Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("loss")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_checkpoint(model, optimizer, scheduler,epoch, train_hist, val_hist, 
                    early_stopper=None, best_val_loss=None,ckpt_path="ckpt.pt"):
    ckpt = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),   # 新增
        "epoch": epoch,
        "train_loss_history": train_hist,
        "val_loss_history": val_hist,
    }
    # 新增：早停状态 + best_val_loss
    if early_stopper is not None:
        ckpt["early_stopper_state"] = {
            "best_loss": early_stopper.best_loss,
            "counter": early_stopper.counter,
        }
    if best_val_loss is not None:
        ckpt["best_val_loss"] = best_val_loss
    torch.save(ckpt, ckpt_path)
    print(f"✅ 保存 checkpoint 到 {ckpt_path}")


def load_checkpoint(model, optimizer, scheduler,ckpt_path="ckpt.pt", device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])  # 新增
    start_epoch = ckpt["epoch"] + 1
    train_hist = ckpt["train_loss_history"]
    #val_hist = ckpt["val_loss_history"]
    val_hist = [v if v is not None else np.nan for v in ckpt["val_loss_history"]]
    
    # 新增：取早停状态（旧 ckpt 没有就返回 None）
    es_state = ckpt.get("early_stopper_state", None)
    best_val_loss = ckpt.get("best_val_loss", None)

    print(f"✅ 加载模型成功，从 epoch {start_epoch} 继续训练")
    return start_epoch, train_hist, val_hist, es_state, best_val_loss

def rel_to_kline(prev_close: float, close_rel: float, high_rel: float, low_rel: float, vol_rel: float):
    """
    输入：上一根收盘价 + 4个rel
    返回：dict K线
    """
    close = prev_close * (1.0 + close_rel)
    high = prev_close * (1.0 + high_rel)
    low = prev_close * (1.0 + low_rel)
    open_ = prev_close  # 当前没有预测open_rel，先用前收代替
    vol = None # vol_rel是相对量比，不是真实成交量，需要单独处理
    # 安全约束：high >= max(open_, close), low <= min(open_, close)
    high = max(high, open_, close)
    low = min(low, open_, close)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vol_rel": vol_rel
    }
