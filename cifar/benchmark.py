import os
import sys
import json
import time
import argparse
import inspect
import pkgutil
import importlib
import uuid
import traceback
from dataclasses import dataclass

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="~/data")
    p.add_argument("--out-dir", type=str, default="workload_dataset")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--all-models", action="store_true", default=False)
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128")
    p.add_argument("--phase", type=str, choices=["train", "infer"], default="train")
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--warmup-steps", type=int, default=30)
    p.add_argument("--measure-steps", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=2)
    pin_group = p.add_mutually_exclusive_group()
    pin_group.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    pin_group.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    dl_group = p.add_mutually_exclusive_group()
    dl_group.add_argument("--download", dest="download", action="store_true")
    dl_group.add_argument("--no-download", dest="download", action="store_false")
    p.set_defaults(download=True)
    p.add_argument("--sample-interval-ms", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--manifest-name", type=str, default="manifest.jsonl")
    p.add_argument("--errors-name", type=str, default="errors.jsonl")
    return p.parse_args()


args0 = parse_args()
PHYSICAL_GPU_ID = int(args0.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU_ID)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from torchvision import transforms

try:
    from pynvml import (
        nvmlInit,
        nvmlShutdown,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetCount,
        NVMLError,
    )
except Exception as e:
    raise RuntimeError("pynvml (nvidia-ml-py) is required. Install: pip install nvidia-ml-py") from e

try:
    import torch.fx as fx
    from torch.fx.passes.shape_prop import ShapeProp
except Exception:
    fx = None
    ShapeProp = None


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_cifar10_loader(batch_size, data_dir, num_workers, pin_memory, phase, download=True):
    root = os.path.expanduser(data_dir)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    is_train = (phase == "train")
    ds = torchvision.datasets.CIFAR10(root=root, train=is_train, download=download, transform=tfm)
    dl = data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=True,
        persistent_workers=(int(num_workers) > 0),
        prefetch_factor=2 if int(num_workers) > 0 else None,
    )
    return dl


def _nanmean(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def _nanmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.nanmax(x))


def _nan_sum_finite(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    m = np.isfinite(x)
    if not m.any():
        return float("nan")
    return float(np.nansum(x[m]))


def _trapz_energy_j(t, p_w):
    t = np.asarray(t, dtype=np.float64)
    p = np.asarray(p_w, dtype=np.float64)
    if t.size < 2 or p.size < 2:
        return float("nan")
    n = min(t.size, p.size)
    t = t[:n]
    p = p[:n]
    dt = np.diff(t)
    if dt.size == 0:
        return float("nan")
    p1 = p[:-1]
    p2 = p[1:]
    m = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(dt) & (dt >= 0.0)
    if not m.any():
        return float("nan")
    e = np.nansum(0.5 * (p1[m] + p2[m]) * dt[m])
    return float(e)


def _dtype_bytes(dtype):
    if dtype is None:
        return 0
    try:
        return int(torch.tensor([], dtype=dtype).element_size())
    except Exception:
        return 0


def _shape_to_4d(shape):
    if shape is None:
        return [-1, -1, -1, -1]
    try:
        s = list(shape)
    except Exception:
        return [-1, -1, -1, -1]
    if len(s) == 0:
        return [-1, -1, -1, -1]
    if len(s) == 1:
        return [int(s[0]), -1, -1, -1]
    if len(s) == 2:
        return [int(s[0]), int(s[1]), -1, -1]
    if len(s) == 3:
        return [int(s[0]), int(s[1]), int(s[2]), -1]
    return [int(s[0]), int(s[1]), int(s[2]), int(s[3])]


def _extract_tensor_meta(obj):
    if obj is None:
        return None
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return obj
    if isinstance(obj, (list, tuple)):
        for z in obj:
            tm = _extract_tensor_meta(z)
            if tm is not None:
                return tm
    if isinstance(obj, dict):
        for _, z in obj.items():
            tm = _extract_tensor_meta(z)
            if tm is not None:
                return tm
    return None


def _tm_shape_dtype(node):
    tm = node.meta.get("tensor_meta", None)
    if tm is None:
        return None, None
    tm1 = _extract_tensor_meta(tm)
    if tm1 is None:
        return None, None
    shp = getattr(tm1, "shape", None)
    dt = getattr(tm1, "dtype", None)
    return shp, dt


def _numel_from_shape(shape):
    if shape is None:
        return None
    try:
        n = 1
        for d in list(shape):
            if d is None:
                return None
            d = int(d)
            if d < 0:
                return None
            n *= d
        return int(n)
    except Exception:
        return None


def fx_graph_or_none(model, example_input):
    if fx is None or ShapeProp is None:
        return None, "torch.fx or ShapeProp unavailable"
    try:
        gm = fx.symbolic_trace(model)
        with torch.no_grad():
            ShapeProp(gm).propagate(example_input)
        modules = dict(gm.named_modules())
        nodes = list(gm.graph.nodes)
        name_to_idx = {n.name: i for i, n in enumerate(nodes)}
        edges = []
        op_vocab = {"placeholder": 0, "get_attr": 1, "call_function": 2, "call_method": 3, "call_module": 4, "output": 5}
        mod_vocab = {"Conv2d": 0, "Linear": 1, "BatchNorm2d": 2, "ReLU": 3, "MaxPool2d": 4, "AvgPool2d": 5, "AdaptiveAvgPool2d": 6, "Dropout": 7, "Other": 8}
        feats = []
        metas = []
        for n in nodes:
            for u in n.all_input_nodes:
                edges.append((name_to_idx[u.name], name_to_idx[n.name]))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if len(edges) else torch.empty((2, 0), dtype=torch.long)
        for n in nodes:
            op = op_vocab.get(n.op, -1)
            op_oh = torch.zeros(len(op_vocab), dtype=torch.float32)
            if op >= 0:
                op_oh[op] = 1.0
            mod_oh = torch.zeros(len(mod_vocab), dtype=torch.float32)
            p = torch.zeros(12, dtype=torch.float32)
            mod_type = "None"
            if n.op == "call_module":
                m = modules.get(n.target, None)
                if m is not None:
                    k = type(m).__name__
                    mod_type = k
                    if k not in mod_vocab:
                        k = "Other"
                    mod_oh[mod_vocab[k]] = 1.0
                    if isinstance(m, nn.Conv2d):
                        p[0] = float(m.in_channels)
                        p[1] = float(m.out_channels)
                        p[2] = float(m.kernel_size[0])
                        p[3] = float(m.kernel_size[1])
                        p[4] = float(m.stride[0])
                        p[5] = float(m.stride[1])
                        p[6] = float(m.padding[0])
                        p[7] = float(m.padding[1])
                        p[8] = float(m.groups)
                    elif isinstance(m, nn.Linear):
                        p[0] = float(m.in_features)
                        p[1] = float(m.out_features)
            out_shape, out_dtype = _tm_shape_dtype(n)
            out4 = _shape_to_4d(out_shape)
            out_numel = _numel_from_shape(out_shape)
            out_bytes = None
            if out_numel is not None:
                out_bytes = int(out_numel * _dtype_bytes(out_dtype))
            in_shape = None
            in_dtype = None
            if len(list(n.all_input_nodes)) > 0:
                in_node = list(n.all_input_nodes)[0]
                in_shape, in_dtype = _tm_shape_dtype(in_node)
            in4 = _shape_to_4d(in_shape)
            flops_est = None
            if n.op == "call_module":
                m = modules.get(n.target, None)
                if m is not None and out_shape is not None:
                    if isinstance(m, nn.Conv2d):
                        try:
                            n_b = int(out_shape[0])
                            c_out = int(out_shape[1])
                            h = int(out_shape[2])
                            w = int(out_shape[3])
                            kh, kw = m.kernel_size
                            cin = m.in_channels
                            g = m.groups
                            macs = n_b * c_out * h * w * (cin // g) * kh * kw
                            flops_est = float(2 * macs)
                        except Exception:
                            flops_est = None
                    elif isinstance(m, nn.Linear):
                        try:
                            n_b = int(out_shape[0])
                            in_f = int(m.in_features)
                            out_f = int(m.out_features)
                            macs = n_b * out_f * in_f
                            flops_est = float(2 * macs)
                        except Exception:
                            flops_est = None
            shape_feat = torch.tensor(in4 + out4, dtype=torch.float32)
            extra_feat = torch.tensor(
                [
                    float(out_numel) if out_numel is not None else -1.0,
                    float(out_bytes) if out_bytes is not None else -1.0,
                    float(flops_est) if flops_est is not None else -1.0,
                ],
                dtype=torch.float32,
            )
            feats.append(torch.cat([op_oh, mod_oh, p, shape_feat, extra_feat], dim=0))
            metas.append(
                {
                    "name": n.name,
                    "op": n.op,
                    "target": str(n.target),
                    "module_type": str(mod_type),
                    "in_shape": list(in_shape) if in_shape is not None else None,
                    "out_shape": list(out_shape) if out_shape is not None else None,
                    "dtype": str(out_dtype) if out_dtype is not None else None,
                    "out_numel": int(out_numel) if out_numel is not None else None,
                    "out_bytes": int(out_bytes) if out_bytes is not None else None,
                    "flops_est": float(flops_est) if flops_est is not None else None,
                }
            )
        x = torch.stack(feats, dim=0)
        return {"edge_index": edge_index.cpu(), "x": x.cpu(), "node_meta": metas}, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)}"


@dataclass
class NVMLSnapshot:
    t: float
    gpu_util: float
    memctrl_util: float
    power_w: float
    vram_used_bytes: float
    vram_total_bytes: float


class NVMLSampler:
    def __init__(self, physical_device_indices, interval_ms=20):
        self.device_indices = list(physical_device_indices)
        self.interval_s = max(1, int(interval_ms)) / 1000.0
        self._stop = False
        self.handles = []
        self.names = []
        self.samples = {i: [] for i in self.device_indices}
        nvmlInit()
        dev_count = nvmlDeviceGetCount()
        for di in self.device_indices:
            if di < 0 or di >= dev_count:
                raise RuntimeError(f"NVML physical GPU index out of range: {di} (device_count={dev_count})")
            h = nvmlDeviceGetHandleByIndex(di)
            self.handles.append(h)
            name = nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            else:
                name = str(name)
            self.names.append(name)

    def start(self):
        import threading
        self._stop = False
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop = True
        if hasattr(self, "_th") and self._th is not None:
            self._th.join()

    def close(self):
        try:
            nvmlShutdown()
        except Exception:
            pass

    def _safe_float(self, v):
        try:
            return float(v)
        except Exception:
            return float("nan")

    def _run(self):
        while not self._stop:
            t = time.time()
            for idx, h in zip(self.device_indices, self.handles):
                gpu_util = float("nan")
                memctrl_util = float("nan")
                p_w = float("nan")
                used_b = float("nan")
                total_b = float("nan")
                try:
                    u = nvmlDeviceGetUtilizationRates(h)
                    gpu_util = self._safe_float(getattr(u, "gpu", float("nan")))
                    memctrl_util = self._safe_float(getattr(u, "memory", float("nan")))
                except NVMLError:
                    pass
                try:
                    pw_mw = nvmlDeviceGetPowerUsage(h)
                    p_w = self._safe_float(pw_mw) / 1000.0
                except NVMLError:
                    pass
                try:
                    mi = nvmlDeviceGetMemoryInfo(h)
                    used_b = self._safe_float(getattr(mi, "used", float("nan")))
                    total_b = self._safe_float(getattr(mi, "total", float("nan")))
                except NVMLError:
                    pass
                self.samples[idx].append(
                    NVMLSnapshot(
                        t=t,
                        gpu_util=gpu_util,
                        memctrl_util=memctrl_util,
                        power_w=p_w,
                        vram_used_bytes=used_b,
                        vram_total_bytes=total_b,
                    )
                )
            time.sleep(self.interval_s)

    def summarize_window(self, t0, t1):
        out = {}
        per_gpu = {}
        for di in self.device_indices:
            s = self.samples.get(di, [])
            if not s:
                per_gpu[di] = {
                    "avg_gpu_util": float("nan"),
                    "avg_memctrl_util": float("nan"),
                    "avg_power": float("nan"),
                    "avg_vram_bytes": float("nan"),
                    "avg_vram_util": float("nan"),
                    "peak_gpu_util": float("nan"),
                    "peak_memctrl_util": float("nan"),
                    "peak_power": float("nan"),
                    "peak_vram_bytes": float("nan"),
                    "peak_vram_util": float("nan"),
                    "energy": float("nan"),
                    "n_samples": 0,
                }
                continue
            ts = np.array([x.t for x in s], dtype=np.float64)
            m = (ts >= t0) & (ts <= t1)
            if not m.any():
                per_gpu[di] = {
                    "avg_gpu_util": float("nan"),
                    "avg_memctrl_util": float("nan"),
                    "avg_power": float("nan"),
                    "avg_vram_bytes": float("nan"),
                    "avg_vram_util": float("nan"),
                    "peak_gpu_util": float("nan"),
                    "peak_memctrl_util": float("nan"),
                    "peak_power": float("nan"),
                    "peak_vram_bytes": float("nan"),
                    "peak_vram_util": float("nan"),
                    "energy": float("nan"),
                    "n_samples": 0,
                }
                continue
            sel = [s[i] for i in np.where(m)[0].tolist()]
            t = np.array([x.t for x in sel], dtype=np.float64)
            gpu_util = np.array([x.gpu_util for x in sel], dtype=np.float64)
            memctrl_util = np.array([x.memctrl_util for x in sel], dtype=np.float64)
            p_w = np.array([x.power_w for x in sel], dtype=np.float64)
            used_b = np.array([x.vram_used_bytes for x in sel], dtype=np.float64)
            total_b = np.array([x.vram_total_bytes for x in sel], dtype=np.float64)
            used_util = np.full_like(used_b, np.nan, dtype=np.float64)
            mm = np.isfinite(used_b) & np.isfinite(total_b) & (total_b > 0)
            used_util[mm] = 100.0 * used_b[mm] / total_b[mm]
            e_j = _trapz_energy_j(t, p_w)
            per_gpu[di] = {
                "avg_gpu_util": _nanmean(gpu_util),
                "avg_memctrl_util": _nanmean(memctrl_util),
                "avg_power": _nanmean(p_w),
                "avg_vram_bytes": _nanmean(used_b),
                "avg_vram_util": _nanmean(used_util),
                "peak_gpu_util": _nanmax(gpu_util),
                "peak_memctrl_util": _nanmax(memctrl_util),
                "peak_power": _nanmax(p_w),
                "peak_vram_bytes": _nanmax(used_b),
                "peak_vram_util": _nanmax(used_util),
                "energy": float(e_j),
                "n_samples": int(len(sel)),
            }
        avg_gpu_utils = [per_gpu[di]["avg_gpu_util"] for di in self.device_indices]
        avg_memctrl_utils = [per_gpu[di]["avg_memctrl_util"] for di in self.device_indices]
        avg_powers = [per_gpu[di]["avg_power"] for di in self.device_indices]
        avg_vram_utils = [per_gpu[di]["avg_vram_util"] for di in self.device_indices]
        avg_vram_bytes = [per_gpu[di]["avg_vram_bytes"] for di in self.device_indices]
        energies = [per_gpu[di]["energy"] for di in self.device_indices]
        out["avg_gpu_util"] = _nanmean(avg_gpu_utils)
        out["avg_memctrl_util"] = _nanmean(avg_memctrl_utils)
        out["avg_power"] = _nanmean(avg_powers)
        out["avg_vram_util"] = _nanmean(avg_vram_utils)
        out["avg_vram_bytes"] = _nanmean(avg_vram_bytes)
        out["energy_sum"] = _nan_sum_finite(energies)
        out["avg_energy_per_gpu"] = _nanmean(energies)
        out["per_gpu"] = per_gpu
        return out


def is_zero_arg_callable(obj):
    if not callable(obj):
        return False
    try:
        sig = inspect.signature(obj)
    except Exception:
        return False
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is inspect._empty:
            return False
    return True


def discover_models():
    import models
    factories = {}
    for m in pkgutil.iter_modules(models.__path__):
        if m.ispkg:
            continue
        mod_name = f"{models.__name__}.{m.name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name, obj in vars(mod).items():
            if name.startswith("_"):
                continue
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                try:
                    inst = obj()
                    if isinstance(inst, nn.Module):
                        factories[name] = (lambda cls=obj: cls())
                except Exception:
                    pass
            elif is_zero_arg_callable(obj):
                try:
                    inst = obj()
                    if isinstance(inst, nn.Module):
                        factories[name] = obj
                except Exception:
                    pass
    return factories


def one_step(model, phase, x, y, criterion, optimizer, scaler, amp):
    if phase == "train":
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=amp):
                _ = model(x)


def run_one(
    model_factory,
    model_key,
    batch_size,
    phase,
    amp,
    warmup_steps,
    measure_steps,
    data_dir,
    num_workers,
    pin_memory,
    out_dir,
    sample_id,
    interval_ms,
    download,
    run_id,
):
    cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU metrics.")
    device = torch.device("cuda:0")
    model = model_factory().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4) if phase == "train" else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if phase == "train" else None
    dl = get_cifar10_loader(batch_size, data_dir, num_workers, pin_memory, phase=phase, download=download)
    it = iter(dl)
    model.train() if phase == "train" else model.eval()
    last_x = None
    last_y = None
    for _ in range(warmup_steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        last_x, last_y = x, y
        if phase == "train":
            one_step(model, phase, x, y, criterion, optimizer, scaler, amp)
        else:
            one_step(model, phase, x, y, criterion, None, None, amp)
    torch.cuda.synchronize()
    graph = None
    graph_error = ""
    if last_x is not None:
        prev_mode = model.training
        model.eval()
        try:
            graph, graph_error = fx_graph_or_none(model, last_x)
        finally:
            model.train(prev_mode)
    sampler = None
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    runtime_gpu_ms = float("nan")
    runtime_wall_ms = float("nan")
    summary = {
        "avg_gpu_util": float("nan"),
        "avg_memctrl_util": float("nan"),
        "avg_power": float("nan"),
        "avg_vram_util": float("nan"),
        "avg_vram_bytes": float("nan"),
        "energy_sum": float("nan"),
        "avg_energy_per_gpu": float("nan"),
        "per_gpu": {},
    }
    gpu_names = []
    t0_wall = None
    t1_wall = None
    try:
        sampler = NVMLSampler(physical_device_indices=[PHYSICAL_GPU_ID], interval_ms=interval_ms)
        gpu_names = sampler.names
        it = iter(dl)
        sampler.start()
        torch.cuda.synchronize()
        t0_wall = time.time()
        starter.record()
        for _ in range(measure_steps):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl)
                x, y = next(it)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if phase == "train":
                one_step(model, phase, x, y, criterion, optimizer, scaler, amp)
            else:
                one_step(model, phase, x, y, criterion, None, None, amp)
        ender.record()
        torch.cuda.synchronize()
        t1_wall = time.time()
        runtime_gpu_ms = float(starter.elapsed_time(ender))
        runtime_wall_ms = float((t1_wall - t0_wall) * 1000.0) if (t0_wall is not None and t1_wall is not None) else float("nan")
        sampler.stop()
        summary = sampler.summarize_window(t0_wall, t1_wall)
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass
            try:
                sampler.close()
            except Exception:
                pass
    per_gpu = summary.get("per_gpu", {})
    per_gpu_ordered = [per_gpu.get(PHYSICAL_GPU_ID, {})]
    sample = {
        "run": {
            "run_id": str(run_id),
            "created_at_unix": float(time.time()),
        },
        "config": {
            "model_key": str(model_key),
            "phase": str(phase),
            "batch_size": int(batch_size),
            "warmup_steps": int(warmup_steps),
            "measure_steps": int(measure_steps),
            "amp": bool(amp),
            "input_shape_chw": [3, 32, 32],
            "interval_ms": int(interval_ms),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "physical_gpu_id": int(PHYSICAL_GPU_ID),
            "data_dir": os.path.expanduser(data_dir),
            "pin_memory": bool(pin_memory),
            "num_workers": int(num_workers),
            "download": bool(download),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
            "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        },
        "label": {
            "runtime_gpu_ms": float(runtime_gpu_ms),
            "runtime_wall_ms": float(runtime_wall_ms),
            "avg_gpu_util": float(summary.get("avg_gpu_util", float("nan"))),
            "avg_memctrl_util": float(summary.get("avg_memctrl_util", float("nan"))),
            "avg_vram_util": float(summary.get("avg_vram_util", float("nan"))),
            "avg_vram_bytes": float(summary.get("avg_vram_bytes", float("nan"))),
            "avg_power": float(summary.get("avg_power", float("nan"))),
            "energy_sum": float(summary.get("energy_sum", float("nan"))),
            "avg_energy_per_gpu": float(summary.get("avg_energy_per_gpu", float("nan"))),
        },
        "detail": {
            "window_wallclock_unix": {"t0": float(t0_wall) if t0_wall is not None else None, "t1": float(t1_wall) if t1_wall is not None else None},
            "per_gpu": per_gpu_ordered,
            "gpu_names": gpu_names,
            "graph_ok": bool(graph is not None),
            "graph_error": str(graph_error) if graph is None else "",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "graph": graph,
    }
    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, f"sample_{sample_id:08d}.pt")
    torch.save(sample, pt_path)
    rec = {
        "run_id": str(run_id),
        "path": pt_path,
        "model": str(model_key),
        "phase": str(phase),
        "batch_size": int(batch_size),
        "runtime_gpu_ms": float(runtime_gpu_ms),
        "runtime_wall_ms": float(runtime_wall_ms),
        "avg_gpu_util": float(summary.get("avg_gpu_util", float("nan"))),
        "avg_memctrl_util": float(summary.get("avg_memctrl_util", float("nan"))),
        "avg_vram_util": float(summary.get("avg_vram_util", float("nan"))),
        "avg_vram_bytes": float(summary.get("avg_vram_bytes", float("nan"))),
        "avg_power": float(summary.get("avg_power", float("nan"))),
        "energy_sum": float(summary.get("energy_sum", float("nan"))),
        "avg_energy_per_gpu": float(summary.get("avg_energy_per_gpu", float("nan"))),
        "graph_ok": bool(graph is not None),
    }
    manifest_path = os.path.join(out_dir, args0.manifest_name)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    run_manifest_path = os.path.join(out_dir, f"manifest_{run_id}.jsonl")
    with open(run_manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _log_error(out_dir, run_id, model_key, phase, batch_size, err_type, err_msg, tb_str):
    os.makedirs(out_dir, exist_ok=True)
    rec = {
        "run_id": str(run_id),
        "time_unix": float(time.time()),
        "model": str(model_key),
        "phase": str(phase),
        "batch_size": int(batch_size),
        "error_type": str(err_type),
        "error_msg": str(err_msg),
        "traceback": str(tb_str),
    }
    path = os.path.join(out_dir, args0.errors_name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    args = args0
    set_seed(args.seed)
    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    bs_list = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]
    if not bs_list:
        raise RuntimeError("No batch sizes provided.")
    factories = discover_models()
    if not factories:
        raise RuntimeError("No instantiable models discovered. Ensure models/__init__.py exists and model constructors are zero-arg compatible.")
    if args.all_models:
        targets = sorted(factories.items(), key=lambda kv: kv[0])
    else:
        if not args.model:
            raise RuntimeError("Provide --model NAME or use --all-models.")
        if args.model not in factories:
            raise RuntimeError(f"--model '{args.model}' not found. Discovered: {sorted(list(factories.keys()))[:80]} ...")
        targets = [(args.model, factories[args.model])]
    os.makedirs(args.out_dir, exist_ok=True)
    meta_path = os.path.join(args.out_dir, f"runmeta_{run_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "created_at_unix": float(time.time()),
                "argv": sys.argv,
                "args": vars(args),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "physical_gpu_id": int(PHYSICAL_GPU_ID),
                "units": {
                    "runtime_gpu_ms": "ms",
                    "runtime_wall_ms": "ms",
                    "avg_gpu_util": "percent(0-100)",
                    "avg_memctrl_util": "percent(0-100)",
                    "avg_vram_util": "percent(0-100)",
                    "avg_vram_bytes": "bytes",
                    "avg_power": "watt",
                    "energy_sum": "joule",
                    "avg_energy_per_gpu": "joule",
                },
                "notes": {
                    "avg_memctrl_util": "NVML utilization.memory (memory controller utilization), NOT VRAM usage ratio",
                    "avg_vram_util": "VRAM used / total ratio derived from NVML memoryInfo",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    sample_id = 0
    for model_key, factory in targets:
        for bs in bs_list:
            sample_id += 1
            try:
                rec = run_one(
                    model_factory=factory,
                    model_key=model_key,
                    batch_size=bs,
                    phase=args.phase,
                    amp=args.amp,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    data_dir=args.data_dir,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    out_dir=args.out_dir,
                    sample_id=sample_id,
                    interval_ms=args.sample_interval_ms,
                    download=args.download,
                    run_id=run_id,
                )
                print(json.dumps(rec, ensure_ascii=False))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                msg = str(e)
                tb = traceback.format_exc()
                if "out of memory" in msg.lower():
                    _log_error(args.out_dir, run_id, model_key, args.phase, bs, type(e).__name__, msg, tb)
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    continue
                _log_error(args.out_dir, run_id, model_key, args.phase, bs, type(e).__name__, msg, tb)
                continue
            except Exception as e:
                msg = str(e)
                tb = traceback.format_exc()
                _log_error(args.out_dir, run_id, model_key, args.phase, bs, type(e).__name__, msg, tb)
                continue
    print(f"saved samples to {args.out_dir}")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()