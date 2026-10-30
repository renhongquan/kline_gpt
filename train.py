import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random
import time
from kline_tokenizer import MultiQuantileTokenizer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from util import *
from kline_model import *


# ====================== 训练 & 评估函数 ======================
def train_one_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    GRAD_NORM_MAX_THRESHOLD = 5.0    # raw norm超过这个值，告警梯度爆炸风险
    GRAD_NORM_MIN_THRESHOLD = 1e-3   # raw norm低于这个值，告警梯度消失风险
    ALERT_WINDOW = 10                # 连续多少个batch触发才告警，避免单点噪声
    grad_norm_history = []
    loader_count = 0
    feat_names = ["close_rel", "high_rel", "low_rel", "vol_rel"]
    n_feat = model.n_feat
    total_sum_loss = 0.0
    feat_loss_sum = [0.0] * n_feat
    feat_cnt = [0] * n_feat
    feat_hard_correct = [0] * n_feat
    feat_soft_correct = [0] * n_feat
    feat_total = [0] * n_feat
    total_step=len(loader)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits_list, loss = model(x, y)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if loader_count % 5 ==0:
            print(f"current step {loader_count}/{total_step} grad : {total_norm}")
        loader_count=loader_count+1
        raw_norm_val = total_norm.item()   
        opt.step()
        total_loss += loss.item()
        grad_norm_history.append(raw_norm_val)
        # 只保留最近ALERT_WINDOW个，防止列表无限变长
        if len(grad_norm_history) > ALERT_WINDOW:
            grad_norm_history.pop(0)

        # 连续窗口判断告警
        if len(grad_norm_history) == ALERT_WINDOW:
            arr = np.array(grad_norm_history)
            avg_norm = arr.mean()

            if avg_norm > GRAD_NORM_MAX_THRESHOLD:
                print(f"⚠️ 【梯度告警】连续{ALERT_WINDOW} batch 梯度norm偏高！avg={avg_norm:.4f}, > {GRAD_NORM_MAX_THRESHOLD}，建议降低LR")
            if avg_norm < GRAD_NORM_MIN_THRESHOLD:
                print(f"⚠️ 【梯度告警】连续{ALERT_WINDOW} batch 梯度norm很低！avg={avg_norm:.6f}, < {GRAD_NORM_MIN_THRESHOLD}，警惕梯度消失")

        for i in range(n_feat):
            logit = logits_list[i]
            y_target = y[..., i]

            feat_loss = F.cross_entropy(
                logit.reshape(-1, model.vocab_size),
                y_target.reshape(-1)
            )
            feat_loss_sum[i] += feat_loss.item()
            feat_cnt[i] += 1

            pred = logit.argmax(dim=-1)
            y_flat = y_target.reshape(-1)
            pred_flat = pred.reshape(-1)

            hard_mask = (pred_flat == y_flat)
            feat_hard_correct[i] += hard_mask.sum().item()

            soft_mask = torch.abs(pred_flat - y_flat) <= 1
            feat_soft_correct[i] += soft_mask.sum().item()

            feat_total[i] += y_flat.size(0)
        
    #t_end = time.time()
    #print(f"    train_one_epoch finished, cost: {t_end - t_start:.2f} s")
    feat_result = {}
    for idx, name in enumerate(feat_names):
        feat_result[name] = {
            "loss": feat_loss_sum[idx] / max(feat_cnt[idx], 1),
            "hard_acc": feat_hard_correct[idx] / max(feat_total[idx], 1),
            "soft_adj_bin": feat_soft_correct[idx] / max(feat_total[idx], 1)
        }

    # 控制台打印详细指标
    print("\n==== Eval Feature Metrics ====")
    for name in feat_names:
        d = feat_result[name]
        print(f"{name:<10s} | loss:{d['loss']:.4f} | hard_acc:{d['hard_acc']:.4f} | soft_adj_acc:{d['soft_adj_bin']:.4f}")
    print("==============================\n")
    return total_loss / len(loader)

@torch.no_grad()
def eval_one_epoch(model, loader, device):
    if len(loader) == 0:
        return float("nan")
    model.eval()
    total_loss = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
    return total_loss / len(loader)





def train_model():
    # -------------------------- 超参配置（对齐你nanochat风格） --------------------------
    SEQ_LEN = 256          # 一次输入多少根K线
    BATCH_SIZE = 32
    LR = 3e-4
    MAX_EPOCH = 500
    EVAL_EVERY = 2       # 每多少迭代评估一次
    SAVE_EVERY = 5
    SPLIT_RATIO_1 = 0.7      # 训练集占比
    SPLIT_RATIO_2 = 0.2      # 验证集占比
    N_BINS = 64
    N_FEAT = 4
    VOCAB_SIZE = N_BINS * N_FEAT
    start_epoch =0

    es_patience = 3
    es_min_delta = 1e-4
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    
    train_loader,val_loader,test_loader =get_dataloader(SEQ_LEN,BATCH_SIZE,SPLIT_RATIO_1,SPLIT_RATIO_2)
       
    print(f"train_loader len:{len(train_loader)}")
    # 模型
    model = KlineGPT_RoPE(
        vocab_size=VOCAB_SIZE,
        n_feat=N_FEAT,
        n_layer=6,
        n_head=8,
        n_embd=256,
        max_kline_seq=SEQ_LEN,
        use_learnable_fuse=True
    ).to(device)

    # 如果你想开启 compile，Windows注意：要装MSVC，否则注释掉
    # model = torch.compile(model, dynamic=False)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95))

    # ========== 学习率调度配置 ==========
    WARMUP_EPOCH = 10       # warmup 5~10 epoch，小数据集用10
    MIN_LR_RATIO = 0.1      # 最终衰减到初始lr的10%
    # 总衰减轮次：总epoch减去warmup轮
    cosine_t_max = MAX_EPOCH - WARMUP_EPOCH

    # 阶段1：warmup 线性上升：0 → LR
    scheduler_warmup = LinearLR(
        optimizer,
        start_factor=1e-8,    # 起始倍率接近0
        end_factor=1.0,       # warmup结束到达目标lr
        total_iters=WARMUP_EPOCH
    )
    # 阶段2：余弦衰减 LR → LR*0.1
    scheduler_cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_t_max,
        eta_min=LR * MIN_LR_RATIO
    )
    # 串联两个调度器
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[WARMUP_EPOCH]   # 第10个epoch之后切换到cosine
    )
  
    best_val_loss = float("inf")
    
    # 训练循环（按迭代数，不是epoch）
    
    train_loss_history = []
    val_loss_history = []
    ckpt_file = "kline_gpt_rope_ckpt.pt"
    best_ckpt_file = "ckpt_best.pt"

    if Path(ckpt_file).exists():
        start_epoch, train_loss_history, val_loss_history,es_state,loaded_best = load_checkpoint(
            model, optimizer, scheduler, ckpt_file, device
        )
        if es_state is not None:
            # 新 ckpt：直接恢复 best_loss 和 counter
            early_stopper.best_loss = es_state["best_loss"]
            early_stopper.counter = es_state["counter"]
            best_val_loss = loaded_best if loaded_best is not None else es_state["best_loss"]
            print(f"🔁 恢复早停状态: best_loss={early_stopper.best_loss:.4f}, "
                f"counter={early_stopper.counter}/{early_stopper.patience}")
        else:
            # 旧 ckpt 兼容：从 history 重算 best_loss，counter 只能从 0 开始
            valid_vals = [v for v in val_loss_history if v is not None and not np.isnan(v)]
            if valid_vals:
                early_stopper.best_loss = min(valid_vals)
                best_val_loss = early_stopper.best_loss
            print(f"⚠️ 旧 ckpt 无 early_stopper_state，counter 从 0 开始")

    for epoch in range(start_epoch, MAX_EPOCH):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        t1 = time.time()
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"current_lr:{current_lr}")
        val_loss = np.nan
        print(f"Epoch {epoch:03d} | train_loss:{train_loss:.4f} | train_cost {t1-t0:.2f}s")
        train_loss_history.append(train_loss)

        # 按EVAL_EVERY执行验证
        if epoch % EVAL_EVERY == 0:
            t0v = time.time()
            val_loss = eval_one_epoch(model, val_loader, device)

            val_loss_history.append(val_loss )
            t1v = time.time()
            print(f"Epoch {epoch:03d} | val_loss:{train_loss:.4f} | val_cost {t1v-t0v:.2f}s")
            # ===== 早停判断放在eval之后 =====
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"\n🛑 Early Stop triggered! patience {es_patience} reached.")
                break
            

            if val_loss < best_val_loss - es_min_delta:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler,epoch, train_loss_history, 
                                val_loss_history, early_stopper=early_stopper,
                                best_val_loss=best_val_loss,ckpt_path=best_ckpt_file)
                print(f"✅ New best val loss {best_val_loss:.4f}, saved ckpt_best.pt")

            
            
        else:
            val_loss_history.append(np.nan)
        
        

        if (epoch % SAVE_EVERY == 0) or (epoch == MAX_EPOCH-1):           
            save_checkpoint(model, optimizer, scheduler,epoch, train_loss_history, 
                            val_loss_history, early_stopper=early_stopper,
                                best_val_loss=best_val_loss,ckpt_path=ckpt_file)
            plot_loss_curve(train_loss_history, val_loss_history, "loss_curve.png")

    print(f"train finished, best val loss: {best_val_loss:.4f}")

    if Path(best_ckpt_file).exists():
        ckpt = torch.load(best_ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print("✅ 加载 best ckpt 用于 test")
        test_loss = eval_one_epoch(model, test_loader, device)
    else:
        print("⚠️ best ckpt 不存在，用当前权重测 test")
        test_loss = eval_one_epoch(model, test_loader, device)

    
    print(f"test loss: {test_loss:.4f}")

def inference_1(device,loader,model,tokenizer):
    # 取一个样本
    
    for x, y, last_close in loader:
        history_tokens = x.to(device)  # (1, seq_len, 4)
        prev_close = last_close.item()
        break

    with torch.no_grad():
        next_token = model.generate_next_kline(history_tokens, temperature=0.9)

    # 预测token
    next_token_np = next_token.cpu().numpy().squeeze(0)
    # 真实标签：y shape [1, 256, 4]，最后一个位置才是下一根K线的真实token
    true_next_token_np = y[:, -1, :].cpu().numpy()
    
    

    pred_local_token_dict = tokenizer.remove_offset(next_token_np)
    true_local_token_dict = tokenizer.remove_offset(true_next_token_np)

    # 解码成rel
    pred_rel_dict = tokenizer.decode_batch(pred_local_token_dict)
    true_rel_dict = tokenizer.decode_batch(true_local_token_dict)


    pred_kline = rel_to_kline(prev_close, 
                              pred_rel_dict["close_rel"][0].item(), 
                              pred_rel_dict["high_rel"][0].item(), 
                              pred_rel_dict["low_rel"][0].item(), 
                              pred_rel_dict["vol_rel"][0].item())
    true_kline = rel_to_kline(prev_close, 
                              true_rel_dict["close_rel"][0].item(),
                              true_rel_dict["high_rel"][0].item(),
                              true_rel_dict["low_rel"][0].item(),
                              true_rel_dict["vol_rel"][0].item(),  
                              )

    # ========== 对比打印 ==========
    print("="*80)
    print(f"{'feature':<12} | {'pred_token':<10} | {'true_token':<10} | {'pred_rel':<12} | {'true_rel':<12}")
    print("-"*80)
    feat_names = ["close_rel", "high_rel", "low_rel", "vol_rel"]
    for i, fname in enumerate(feat_names):
        pt = next_token_np[0, i]
        tt = true_next_token_np[0, i]
        pr = pred_rel_dict[fname][0].item()
        tr = true_rel_dict[fname][0].item()
        print(f"{fname:<12} | {pt:<10d} | {tt:<10d} | {pr:<12.6f} | {tr:<12.6f}")
    print("="*80)
    print(f"【预测K线】 {pred_kline}")
    print(f"【真实K线】 {true_kline}")
    print("="*80)


def inference(data_npy:str,close_npy:str,kline_bounds:str ="kline_bounds.json",kline_gpt_ckpt :str ="kline_gpt_rope_ckpt.pt"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE=256
    N_FEAT=4
    SEQ_LEN=256
    N_LAYER=6
    N_HEAD=8
    N_EMBD=256
    BATCH_SIZE=1
    SPLIT_RATIO_1=0.8
    SPLIT_RATIO_2=0.1
    dropout_rate=0.1
    
    tokenizer = MultiQuantileTokenizer.load(kline_bounds)
    
    model = KlineGPT_RoPE(
            vocab_size=VOCAB_SIZE,
            n_feat=N_FEAT,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            n_embd=N_EMBD,
            max_kline_seq=SEQ_LEN,
            use_learnable_fuse=True,
            dropout_rate=dropout_rate
        ).to(device)
    ckpt = torch.load(kline_gpt_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    train_loader,val_loader,test_loader =get_dataloader(SEQ_LEN,BATCH_SIZE,SPLIT_RATIO_1,SPLIT_RATIO_2,data_npy,close_npy)
    loads= [train_loader,val_loader,test_loader]

    for load in loads:
        inference_1(device,load,model,tokenizer)


    
    

    

if __name__ == "__main__":
    SEED = 42
    data_npy = "stacked_tokens.npy"
    close_npy = "close_ref.npy"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    train_model() 
    inference(data_npy,close_npy)   
