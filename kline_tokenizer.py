import json
import numpy as np
import sys
from pathlib import Path
import pandas as pd


class SingleQuantileTokenizer:
    def __init__(self, n_bins: int = 64, q_low: float = 0.0, q_high: float = 1.0):
        self.n_bins = n_bins
        self.q_low = q_low
        self.q_high = q_high
        self.boundaries = None  # [n_bins + 1]
        self.bin_centers = None # 桶中点，用于解码还原

    def fit(self, data: np.ndarray):
        """仅在训练集上拟合边界，输入一维数组"""
        if data.ndim != 1:
            raise ValueError("data must be 1d array")
        data = data[np.isfinite(data)]
        if len(data) == 0:
            raise RuntimeError("fit error: after filter nan/inf, no valid data left!")
        qs = np.linspace(self.q_low, self.q_high, self.n_bins + 1)
        self.boundaries = np.quantile(data, qs)
        self.bin_centers = (self.boundaries[:-1] + self.boundaries[1:]) / 2.0

    def encode(self, x: np.ndarray) -> np.ndarray:
        """向量化编码，支持任意shape: (T,) / (B, T)"""
        if self.boundaries is None:
            raise RuntimeError("Tokenizer not fitted!")
        tokens = np.digitize(x, self.boundaries) - 1
        tokens = np.clip(tokens, 0, self.n_bins - 1)
        return tokens.astype(np.int64)

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        """token → 桶中点还原相对值"""
        if self.bin_centers is None:
            raise RuntimeError("Tokenizer not fitted!")
        tokens = np.clip(tokens, 0, self.n_bins - 1)
        return self.bin_centers[tokens]

    def to_dict(self):
        return {
            "n_bins": self.n_bins,
            "q_low": self.q_low,
            "q_high": self.q_high,
            "boundaries": self.boundaries.tolist(),
            "bin_centers": self.bin_centers.tolist()
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["n_bins"], d["q_low"], d["q_high"])
        obj.boundaries = np.array(d["boundaries"])
        obj.bin_centers = np.array(d["bin_centers"])
        return obj


class MultiQuantileTokenizer:
    def __init__(self, feature_names: list[str], n_bins: int = 64):
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.tokenizers: dict[str, SingleQuantileTokenizer] = {
            name: SingleQuantileTokenizer(n_bins) for name in feature_names
        }
        self.offsets = {name: i * n_bins for i, name in enumerate(feature_names)}
        self.vocab_size = len(feature_names) * n_bins

    def fit(self, data_dict: dict[str, np.ndarray]):
        """data_dict: {feat_name: 一维序列}"""
        for fname, arr in data_dict.items():
            self.tokenizers[fname].fit(arr)

    def encode_batch(self, batch_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        批量向量化编码
        输入：{"open_rel": array, "high_rel": array, ...}  shape 支持 (T,) / (B, T)
        输出：每个特征独立局部token [0, n_bins-1]，不带offset
        """
        out = {}
        for fname, arr in batch_data.items():
            out[fname] = self.tokenizers[fname].encode(arr)
        return out

    def decode_batch(self, token_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """批量解码：局部token → 还原rel相对值"""
        out = {}
        for fname, tokens in token_dict.items():
            out[fname] = self.tokenizers[fname].decode(tokens)
        return out

    def add_offset(self, token_dict: dict[str, np.ndarray]) -> np.ndarray:
        """
        把分开的各特征局部token，加上offset拼成一条序列
        输入：{open_rel:(B,T), high_rel:(B,T), ...}
        输出：(B, T, n_feat)，或者展平成 (B, T*n_feat) 按需自己改
        """
        stack = []
        for fname in self.feature_names:
            stack.append(token_dict[fname] + self.offsets[fname])
        return np.stack(stack, axis=-1)

    def remove_offset(self, stacked_tokens: np.ndarray) -> dict[str, np.ndarray]:
        """
        从加完offset的堆叠token，还原成每个特征独立局部token
        stacked_tokens shape: (B, T, n_feat) 或者 (B, n_feat)
        """
        out = {}
        for idx, fname in enumerate(self.feature_names):
            local_tokens = stacked_tokens[..., idx] - self.offsets[fname]
            out[fname] = np.clip(local_tokens, 0, self.n_bins - 1)
        return out

    def dump(self, path: str):
        payload = {            
            "feature_names": self.feature_names,
            "n_bins": self.n_bins,
            "offsets": self.offsets,   
            "tokenizers": {k: v.to_dict() for k, v in self.tokenizers.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        obj = cls(payload["feature_names"], payload["n_bins"])
        for k, v in payload["tokenizers"].items():
            obj.tokenizers[k] = SingleQuantileTokenizer.from_dict(v)

        if "offsets" in payload:
            # 新版json：直接读取保存好的offsets
            obj.offsets = payload["offsets"]
            print(f"✅ load tokenizer: use saved offsets from json")
        else:
            # 旧版json没有offsets，现场计算，做向后兼容
            print(f"⚠️ old json file: offsets not found, recalc by feature_names order")
            obj.offsets = {name: i * obj.n_bins for i, name in enumerate(obj.feature_names)}

        # 防御校验
        assert set(obj.offsets.keys()) == set(obj.feature_names), \
            f"offsets keys mismatch! offsets keys:{list(obj.offsets.keys())}, feature_names:{obj.feature_names}"

        obj.vocab_size = len(obj.feature_names) * obj.n_bins
        return obj

def rel_to_price(rel_dict: dict[str, np.ndarray], prev_base_price: float):
    out = {}
    for k, rel in rel_dict.items():
        out[k] = prev_base_price * (1.0 + rel)
    return out

def pre_data():
    # --------配置区--------
    CSV_PATH = "./finetune_csv/data/btc_1d.csv"
    OUT_NPY = "stacked_tokens_1d.npy"
    OUT_CLOSE_NPY = "close_ref_1d.npy"
    OUT_JSON = "kline_bounds_1d.json"

    # 和你代码保持一致的4个特征
    FEATURES = ["close_rel", "high_rel", "low_rel", "vol_rel"]
    N_BINS = 64
    TRAIN_SPLIT_RATIO = 0.9   # 前90%时序用来fit tokenizer，防泄露

    # 1.读取K线，时间排序
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("open_time").reset_index(drop=True)
    print(f"原始K线行数: {len(df)}")

    # 2.计算特征（参考你示例的计算方式：用上一根close做基准）
    df["close_rel"] = (df["close"] - df["close"].shift(1)) / df["close"].shift(1)
    df["high_rel"]  = (df["high"]  - df["close"].shift(1)) / df["close"].shift(1)
    df["low_rel"]   = (df["low"]   - df["close"].shift(1)) / df["close"].shift(1)
    df["vol_rel"]   = df["volume"] / df["volume"].shift(1).rolling(window=20).mean()

    # 删掉NaN（shift产生第一行nan、rolling窗口nan）
    df = df.dropna().reset_index(drop=True)
    print(f"过滤NaN后有效行数: {len(df)}")

    # 过滤正负无穷inf
    mask_inf = ~np.isinf(df[["close_rel","high_rel","low_rel","vol_rel"]]).any(axis=1)
    df = df.loc[mask_inf].reset_index(drop=True)
    print(f"过滤inf后有效行数: {len(df)}")

    # 构造完整特征字典，全部时序
    full_data_dict = {
        "close_rel": df["close_rel"].to_numpy(copy=True),
        "high_rel":  df["high_rel"].to_numpy(copy=True),
        "low_rel":   df["low_rel"].to_numpy(copy=True),
        "vol_rel":   df["vol_rel"].to_numpy(copy=True),
    }

    n_total = len(full_data_dict["close_rel"])
    n_train_fit = int(n_total * TRAIN_SPLIT_RATIO)

    # 【关键！fit只拿前面训练片段，不要拿全部数据】
    fit_data_dict = {k: v[:n_train_fit] for k, v in full_data_dict.items()}

    # 3.初始化tokenizer，fit训练片段
    mt = MultiQuantileTokenizer(feature_names=FEATURES, n_bins=N_BINS)
    mt.fit(fit_data_dict)
    mt.dump(OUT_JSON)
    print(f"已保存tokenizer配置 -> {OUT_JSON}")

    # 4.对【全部时序数据】做encode
    token_dict = mt.encode_batch(full_data_dict)

    # 5.各特征加上offset，拼成 [N,4]
    stacked_tokens = mt.add_offset(token_dict)
    print(f"stacked_tokens shape: {stacked_tokens.shape} len:{len(stacked_tokens)}")
    print(f"dtype: {stacked_tokens.dtype}")
    print(f"token min={stacked_tokens.min()}, max={stacked_tokens.max()}")

    # 6.保存npy，给训练脚本使用
    np.save(OUT_NPY, stacked_tokens)
    print(f"保存完成：{OUT_NPY}")
    close_ref_arr = df["close"].to_numpy(dtype=np.float32)
    np.save(OUT_CLOSE_NPY, close_ref_arr)
    print(f"saved close_ref.npy, shape: {close_ref_arr.shape} len: {len(close_ref_arr)}")
        
if __name__ == "__main__":
    pre_data()