"""
This script provides an exmaple to wrap UER-py for classification.
"""
import random
import argparse
import math
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
from torch import optim
import time
import os
import torch.nn.functional as F
import numpy as np
from datetime import timedelta 
import psutil
import json
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
from scipy.stats import pearsonr, spearmanr


print("Number of GPUs:", torch.cuda.device_count())
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"

class TrainingEfficiencyMonitor:
    """
    Training efficiency monitor for comparing different training methods
    Collects key efficiency metrics and saves the results
    """

    def __init__(self,
                 experiment_name: str,
                 method_name: str,
                 output_dir: str = "./efficiency_results_app",
                 monitor_interval: float = 1.0):
        """
        Initialize the monitor

        Args:
            experiment_name: Experiment name
            method_name: Method name
            output_dir: Output directory for results
            monitor_interval: Monitoring interval (seconds)
        """
        self.experiment_name = experiment_name
        self.method_name = method_name
        self.output_dir = output_dir
        self.monitor_interval = monitor_interval

        
        os.makedirs(output_dir, exist_ok=True)

        
        self.metrics_history = []
        self.start_time = None
        self.process = psutil.Process()  

        print(f"🔍 Start monitoring: {experiment_name} - {method_name}")

    def start(self):
        """Start monitoring"""
        self.start_time = datetime.now()
        initial_metrics = self._collect_metrics("initial")
        self.metrics_history.append(initial_metrics)

        print(f"⏱️  Start time: {self.start_time.strftime('%H:%M:%S')}")
        return self

    def _collect_metrics(self, phase: str) -> Dict:
        """Collect key efficiency metrics"""
        try:
            with self.process.oneshot():
                mem_info = self.process.memory_info()

                return {
                    'timestamp': datetime.now().isoformat(),
                    'phase': phase,
                    'runtime_seconds': (datetime.now() - self.start_time).total_seconds(),

                    
                    'memory_mb': mem_info.rss / 1024 / 1024,  
                    'cpu_percent': self.process.cpu_percent(interval=None),  
                    'threads': self.process.num_threads(),  

                    
                    'io_read_mb': self.process.io_counters().read_bytes / 1024 / 1024 if hasattr(self.process, 'io_counters') else 0,
                }
        except:
            return None

    def record_checkpoint(self, checkpoint_name: str):
        """Record a checkpoint (e.g., at the end of each epoch)"""
        metrics = self._collect_metrics(f"checkpoint_{checkpoint_name}")
        if metrics:
            self.metrics_history.append(metrics)
        return metrics

    def stop(self, final_stats: Dict = None) -> Dict:
        """Stop monitoring and return the efficiency summary"""
        if not self.start_time:
            return {}

        
        final_metrics = self._collect_metrics("final")
        if final_metrics:
            self.metrics_history.append(final_metrics)

        
        training_metrics = [
            m for m in self.metrics_history
            if m and m.get('phase', '').startswith('checkpoint') or m.get('phase') == 'training'
        ]

        
        efficiency_summary = self._calculate_efficiency_summary(training_metrics)

        
        if final_stats:
            efficiency_summary.update(final_stats)

        
        self._save_results(efficiency_summary)

        return efficiency_summary

    def _calculate_efficiency_summary(self, training_metrics: List[Dict]) -> Dict:
        """Compute the efficiency summary using only the key metrics"""
        if not training_metrics:
            return {}

        
        memory_values = [m['memory_mb'] for m in training_metrics]
        cpu_values = [m['cpu_percent'] for m in training_metrics]
        thread_values = [m['threads'] for m in training_metrics]

        
        initial_memory = self.metrics_history[0]['memory_mb'] if self.metrics_history else 0
        final_memory = self.metrics_history[-1]['memory_mb'] if self.metrics_history else 0

        return {
            
            'experiment': self.experiment_name,
            'method': self.method_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),

            
            'memory_initial_mb': initial_memory,
            'memory_final_mb': final_memory,
            'memory_peak_mb': max(memory_values) if memory_values else 0,
            'memory_avg_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
            'memory_growth_mb': final_memory - initial_memory,

            
            'cpu_peak_percent': max(cpu_values) if cpu_values else 0,
            'cpu_avg_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,

            
            'threads_avg': sum(thread_values) / len(thread_values) if thread_values else 0,
            'threads_max': max(thread_values) if thread_values else 0,

            
            'efficiency_score': self._calculate_efficiency_score(
                sum(memory_values) / len(memory_values) if memory_values else 0,
                sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                final_memory - initial_memory
            )
        }

    def _calculate_efficiency_score(self, avg_memory: float, avg_cpu: float, memory_growth: float) -> float:
        """Compute the overall efficiency score

        Weight allocation:
        - Memory usage: 40%
        - CPU usage: 30%
        - Memory growth: 30%

        A lower score indicates higher efficiency
        """
        
        memory_score = min(avg_memory / 1000, 1.0)  
        cpu_score = min(avg_cpu / 100, 1.0)  
        growth_score = min(abs(memory_growth) / 500, 1.0)  

        
        efficiency = (memory_score * 0.4) + (cpu_score * 0.3) + (growth_score * 0.3)

        return round(efficiency, 3)

    def _save_results(self, efficiency_summary: Dict):
        """Save monitoring results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{self.experiment_name}_{self.method_name}"

        
        summary_file = os.path.join(self.output_dir, f"{base_filename}_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(efficiency_summary, f, indent=2, ensure_ascii=False)

        
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            history_file = os.path.join(self.output_dir, f"{base_filename}_history_{timestamp}.csv")
            df.to_csv(history_file, index=False, encoding='utf-8')

        
        self._update_comparison_file(efficiency_summary)

        
        self._print_summary(efficiency_summary)

        print(f"\n💾 Results saved to: {self.output_dir}/")
        print(f"  Efficiency summary: {os.path.basename(summary_file)}")
        if self.metrics_history:
            print(f"  Detailed data: {os.path.basename(history_file)}")

    def _update_comparison_file(self, efficiency_summary: Dict):
        """Update the method comparison file"""
        comparison_file = os.path.join(self.output_dir, f"{self.experiment_name}_comparison.csv")

        
        comparison_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': efficiency_summary['method'],
            'duration_seconds': efficiency_summary['duration_seconds'],
            'memory_peak_mb': efficiency_summary['memory_peak_mb'],
            'memory_avg_mb': efficiency_summary['memory_avg_mb'],
            'memory_growth_mb': efficiency_summary['memory_growth_mb'],
            'cpu_avg_percent': efficiency_summary['cpu_avg_percent'],
            'efficiency_score': efficiency_summary['efficiency_score']
        }

        
        df_new = pd.DataFrame([comparison_data])

        
        if os.path.exists(comparison_file):
            df_existing = pd.read_csv(comparison_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        
        df_combined.to_csv(comparison_file, index=False, encoding='utf-8')

    def _print_summary(self, efficiency_summary: Dict):
        """Print the efficiency summary"""
        print(f"\n{'='*60}")
        print(f"📊 Training efficiency report: {efficiency_summary['experiment']} - {efficiency_summary['method']}")
        print(f"{'='*60}")

        print(f"⏱️  Training duration: {efficiency_summary['duration_seconds']:.1f}s")

        print(f"\n💾 Memory usage:")
        print(f"  Initial: {efficiency_summary['memory_initial_mb']:.1f} MB")
        print(f"  Final: {efficiency_summary['memory_final_mb']:.1f} MB")
        print(f"  Peak: {efficiency_summary['memory_peak_mb']:.1f} MB")
        print(f"  Average: {efficiency_summary['memory_avg_mb']:.1f} MB")
        print(f"  Growth: {efficiency_summary['memory_growth_mb']:+.1f} MB")

        print(f"\n⚡ CPU usage:")
        print(f"  Peak: {efficiency_summary['cpu_peak_percent']:.1f}%")
        print(f"  Average: {efficiency_summary['cpu_avg_percent']:.1f}%")

        print(f"\n🧵 Thread usage:")
        print(f"  Average: {efficiency_summary['threads_avg']:.1f}")
        print(f"  Maximum: {efficiency_summary['threads_max']}")

        print(f"\n⭐ Overall efficiency score: {efficiency_summary['efficiency_score']} (lower is better)")
        print(f"{'='*60}")
class SurrogateLossPredictor(nn.Module):
    """
    Lightweight surrogate model that predicts the loss for given (src, seg, tgt, soft_tgt, soft_alpha, soft_targets)
    inputs.
    """
    def __init__(self, args):
        super(SurrogateLossPredictor, self).__init__()

        self.emb_size = args.emb_size
        self.labels_num = args.labels_num
        
        self.loss_hidden_size = args.loss_hidden_size
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        
        
        self.dropout = nn.Dropout(args.dropout)

        
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.emb_size, self.loss_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.loss_hidden_size, self.loss_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

        
        self.target_embedding = nn.Embedding(self.labels_num, self.loss_hidden_size)

        
        self.soft_tgt_projector = nn.Sequential(
            nn.Linear(self.labels_num, self.loss_hidden_size),
            nn.ReLU()
        )

        
        self.num_hyper_features = 2

        
        
        if self.soft_targets:
            final_feature_dim = (self.loss_hidden_size * 3) + self.num_hyper_features
        else:
            final_feature_dim = self.loss_hidden_size * 2
        self.loss_predictor = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(final_feature_dim//2, final_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(final_feature_dim // 2, 1) 
        )

    def forward(self, emb, tgt, soft_tgt):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
            tgt: [batch_size]
            soft_tgt: [batch_size x labels_num] (Logits/Probabilities)
            soft_alpha: [batch_size] (floating-point weight)
            soft_targets: [batch_size] (Boolean value represented as floating-point 0.0 or 1.0)

        Returns:
            predicted_loss: [batch_size x 1]
        """

        
        
        features = self.feature_extractor(emb)
        pooled_features = torch.mean(features, dim=(1,2)) # Global Mean Pooling
        pooled_features = self.dropout(pooled_features)
        
        
        tgt_emb = self.target_embedding(tgt.view(-1))
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                soft_tgt_features = self.soft_tgt_projector(soft_tgt)
                batch_size = soft_tgt_features.size(0)
                soft_alpha_feature = torch.full(
                    (batch_size, 1), float(self.soft_alpha),
                    device=soft_tgt_features.device, dtype=soft_tgt_features.dtype
                )
                soft_targets_feature = torch.ones(
                    (batch_size, 1), device=soft_tgt_features.device, dtype=soft_tgt_features.dtype
                )
                hyper_features = torch.cat((soft_alpha_feature, soft_targets_feature), dim=-1)
                combined_features = torch.cat((pooled_features, tgt_emb, soft_tgt_features, hyper_features), dim=-1)
                predicted_loss = self.loss_predictor(combined_features)
            else:
                combined_features = torch.cat((pooled_features, tgt_emb), dim=-1)
                predicted_loss = self.loss_predictor(combined_features)
        return predicted_loss

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.method = args.method
        self.focal_gamma = getattr(args, "focal_gamma", 2.0)
        self.focal_alpha = getattr(args, "focal_alpha", 1.0)
        if self.method == "TDRO":
            self.tdro_alpha = nn.Parameter(torch.zeros(()))
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def _hard_label_loss(self, logits, tgt):
        log_probs = F.log_softmax(logits, dim=1)
        per_sample_ce = -log_probs.gather(dim=1, index=tgt.unsqueeze(1)).squeeze(1)
        if self.method != "Focal":
            return per_sample_ce
        probs = log_probs.exp()
        pt = probs.gather(dim=1, index=tgt.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        focal_weight = (1.0 - pt).pow(self.focal_gamma)
        return self.focal_alpha * focal_weight * per_sample_ce

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        ### src,seg [bz x 640] -> [bz x 5 x 128], set seq_length = 128
        batch_size_num = src.shape[0]
        seq_length = src.shape[1] // 5  
        emb_data = self.embedding(src, seg)  

        
        output = torch.Tensor(0).to(src.device)  
        for each_batch_size in range(emb_data.size(0)):
            emb = emb_data[each_batch_size]  # [5 x seq_length x 768]
            seg_data = seg[each_batch_size]  # [5 x seq_length]
            output_emb = self.encoder(emb, seg_data)  # [5 x seq_length x 768]
            output_data = output_emb[:, :1, :]  
            cls_output = output_data.squeeze(1).unsqueeze(0)  # [1 x 5 x 768]
            if output.size(0) == 0:
                output = cls_output
            else:
                output = torch.cat((output, cls_output), 0)  

        
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)  # [batch_size x 768]
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]  # [batch_size x 768]
        elif self.pooling == "last":
            output = output[:, -1, :]  # [batch_size x 768]
        else:  
            output = output[:, 0, :]  # [batch_size x 768]

        
        output = torch.tanh(self.output_layer_1(output))  # [batch_size x hidden_dim]
        logits = self.output_layer_2(output)  

        
        loss = None
        per_sample_loss = None  

        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                
                
                mse_loss_per_element = F.mse_loss(logits, soft_tgt, reduction='none')  # [batch_size x output_dim]
                per_sample_mse = mse_loss_per_element.sum(dim=1)  

                
                log_probs = F.log_softmax(logits, dim=1)  # [batch_size x output_dim]
                
                per_sample_nll = -log_probs.gather(
                    dim=1,
                    index=tgt.unsqueeze(1)
                ).squeeze(1)
                if self.method == "Focal":
                    per_sample_nll = self._hard_label_loss(logits, tgt)

                per_sample_loss = self.soft_alpha * per_sample_mse + (1 - self.soft_alpha) * per_sample_nll
                loss = per_sample_loss.mean()  

            else:
                
                log_probs = F.log_softmax(logits, dim=1)  # [batch_size x output_dim]
                per_sample_loss = -log_probs.gather(
                    dim=1,
                    index=tgt.unsqueeze(1)
                ).squeeze(1)
                if self.method == "Focal":
                    per_sample_loss = self._hard_label_loss(logits, tgt)
                loss = per_sample_loss.mean()

        
        return loss, per_sample_loss, logits


def compute_pearson_corr(pred_loss, true_loss):
    if len(pred_loss) < 2:
        return {"pearson_r": 0.0, "p_value": 1.0, "significant": False}
    corr, p_value = pearsonr(pred_loss, true_loss)
    corr = 0.0 if np.isnan(corr) else float(corr)
    p_value = 1.0 if np.isnan(p_value) else float(p_value)
    return {"pearson_r": corr, "p_value": p_value, "significant": p_value < 0.05}


def compute_spearman_corr(pred_loss, true_loss):
    if len(pred_loss) < 2:
        return {"spearman_rho": 0.0, "p_value": 1.0}
    corr, p_value = spearmanr(pred_loss, true_loss)
    return {
        "spearman_rho": float(corr) if not np.isnan(corr) else 0.0,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
    }


def precision_at_k(pred_scores, true_losses, k=32):
    k = max(1, min(int(k), len(pred_scores), len(true_losses)))
    top_k_pred = set(np.argsort(pred_scores)[-k:])
    top_k_true = set(np.argsort(true_losses)[-k:])
    return len(top_k_pred & top_k_true) / k


def get_tdro_alpha(model):
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "tdro_alpha"):
        raise AttributeError("TDRO requires Classifier.tdro_alpha.")
    return module.tdro_alpha


def tdro_softplus_loss(per_sample_loss, alpha, rho, lam):
    rho = float(rho)
    lam = float(lam)
    if not (0.0 < rho < 1.0):
        raise ValueError("--tdro_rho must be in (0, 1).")
    if lam <= 0.0:
        raise ValueError("--tdro_lambda must be positive.")
    losses = per_sample_loss.view(-1)
    exponent = (losses - alpha) / lam + math.log(rho)
    return alpha + (lam / rho) * F.softplus(exponent).mean()


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location='cpu', weights_only=True), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                    scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}

    with open(path, mode="r", encoding="utf-8") as f:
        try:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    for i, column_name in enumerate(line.strip().split("\t")):
                        columns[column_name] = i
                    continue
                line = line[:-1].split("\t")
                tgt = int(line[columns["label"]])
                if args.soft_targets and "logits" in columns.keys():
                    soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]

                src_dataset,seg_dataset = [], [] # not source code
                if "text_b" not in columns:  # Sentence classification.
                    text_a = line[columns["text_a"]]
                    ### source code as up
                    text_a_list = text_a.split(" | ")
                    if text_a_list:
                        for text_a_index in range(len(text_a_list)):
                            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a_list[text_a_index]))
                            src_dataset.append(src)
                            seg_dataset.append([1] * len(src))
                    else:
                        print("BBBB ",text_a_list," BBBBBBBBBBBBB",path)
                ### source codes as below
                
                
                else:  # Sentence-pair classification.
                    text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                    src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                    src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                if src_dataset:
                    for index in range(len(src_dataset)):
                        if len(src_dataset[index]) > args.seq_length:
                            src_dataset[index] = src_dataset[index][: args.seq_length]
                            seg_dataset[index] = seg_dataset[index][: args.seq_length]
                        while len(src_dataset[index]) < args.seq_length:
                            src_dataset[index].append(0)
                            seg_dataset[index].append(0)
                else:
                    print("BBBB ",text_a_list," BBBBBBBBBBBBB",path)
                # src_dataset,seg_dataset [5 x 128] -> src,seg [640]
                
                
                src = src_dataset 
                seg = seg_dataset

                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, soft_tgt))
                else:
                    dataset.append((src, tgt, seg))
        except Exception as e:
            print(path)
            print(e)

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None,inference=False):
    if inference:
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        if soft_tgt_batch is not None:
            soft_tgt_batch = soft_tgt_batch.to(args.device)

        loss, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)
            loss_ = loss_.view(-1)
    else:
        model.zero_grad()

        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        if soft_tgt_batch is not None:
            soft_tgt_batch = soft_tgt_batch.to(args.device)

        loss, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)

            loss_ = loss_.view(-1)
        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()

    return loss,loss_


def _metrics_from_confusion(confusion):
    """Return per-class and macro precision/recall/F1 with zero-denominator protection.

    Confusion matrix convention in this script is [predicted_label, gold_label].
    """
    labels_num = confusion.size(0)
    rem = np.zeros((labels_num, 4), dtype=np.float64)
    precision_list, recall_list, f1_list = [], [], []

    for label in range(labels_num):
        tp = confusion[label, label].item()
        pred_count = confusion[label, :].sum().item()
        gold_count = confusion[:, label].sum().item()

        precision = tp / pred_count if pred_count > 0 else 0.0
        recall = tp / gold_count if gold_count > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rem[label] = [label, precision, recall, f1]
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return rem, float(np.mean(precision_list)), float(np.mean(recall_list)), float(np.mean(f1_list))


def _build_confusion(pred, gold, labels_num):
    confusion = torch.zeros(labels_num, labels_num, dtype=torch.long)
    for p, g in zip(pred.tolist(), gold.tolist()):
        confusion[p, g] += 1
    return confusion


def _append_results_summary(args, dataset_name, metrics_by_alpha, summary_path="./CIKM_all_results_summary.csv"):
    """Append one row per alpha for the completed (dataset, method, seed) test run."""
    rows = []
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for alpha, metrics in metrics_by_alpha.items():
        rows.append({
            'timestamp': timestamp,
            'dataset': dataset_name,
            'method': args.method,
            'seed': args.seed,
            'alpha': alpha,
            'tail_size': metrics['tail_size'],
            'CVaR': metrics['CVaR'],
            'AC': metrics['accuracy'],
            'PR': metrics['macro_precision'],
            'RC': metrics['macro_recall'],
            'F1': metrics['macro_f1'],
        })

    df = pd.DataFrame(rows)
    write_header = not os.path.exists(summary_path)
    df.to_csv(summary_path, mode='a', header=write_header, index=False, encoding='utf-8')
    print(f"Appended test results to {summary_path}")


def evaluate(args, dataset, print_confusion_matrix=False, save_path=None, dataset_name=None):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    args.model.eval()

    # Run inference exactly once so that loss, prediction, gold label, and sample index
    # remain perfectly aligned.  All tail metrics are then computed from exact top-k indices.
    loss_chunks, pred_chunks, gold_chunks = [], [], []
    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, loss_, logits = args.model(src_batch, tgt_batch, seg_batch)

        loss_chunks.append(loss_.view(-1).detach().cpu())
        pred_chunks.append(torch.argmax(logits, dim=1).view(-1).detach().cpu())
        gold_chunks.append(tgt_batch.view(-1).detach().cpu())
        print(f"c:{i}/{args.train_steps}")

    if len(loss_chunks) == 0:
        raise ValueError("evaluate() received an empty dataset.")

    loss_list = torch.cat(loss_chunks, dim=0)
    pred_all = torch.cat(pred_chunks, dim=0)
    gold_all = torch.cat(gold_chunks, dim=0)
    sample_count = len(loss_list)

    confusion = _build_confusion(pred_all, gold_all, args.labels_num)
    correct = int((pred_all == gold_all).sum().item())
    accuracy = correct / sample_count
    rem, macro_precision, macro_recall, macro_f1 = _metrics_from_confusion(confusion)

    alpha_list = [0.0, 0.5, 0.7, 0.9]
    CVaR_list, acc_list = [], []
    tail_confusions = {}
    metrics_by_alpha = {}
    tail_membership = {}

    for alpha in alpha_list:
        # Keep the original definition k = int(N * (1-alpha)), but select exactly k
        # samples using topk indices.  This avoids threshold-tie over-selection.
        k = max(1, int(sample_count * (1.0 - alpha)))
        loss_alpha, tail_indices = torch.topk(loss_list, k=k, largest=True, sorted=True)
        tail_pred = pred_all[tail_indices]
        tail_gold = gold_all[tail_indices]

        tail_confusion = _build_confusion(tail_pred, tail_gold, args.labels_num)
        tail_rem, tail_pr, tail_rc, tail_f1 = _metrics_from_confusion(tail_confusion)
        tail_acc = float((tail_pred == tail_gold).sum().item() / k)
        cvar = float(loss_alpha.mean().item())

        CVaR_list.append(cvar)
        acc_list.append(tail_acc)
        tail_confusions[alpha] = tail_confusion
        metrics_by_alpha[alpha] = {
            'tail_size': k,
            'CVaR': cvar,
            'accuracy': tail_acc,
            'macro_precision': tail_pr,
            'macro_recall': tail_rc,
            'macro_f1': tail_f1,
            'per_class': tail_rem,
        }

        membership = np.zeros(sample_count, dtype=np.int8)
        membership[tail_indices.numpy()] = 1
        tail_membership[alpha] = membership

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        for label in range(confusion.size(0)):
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(
                label, rem[label, 1], rem[label, 2], rem[label, 3]))
        print("Macro PR/RC/F1: {:.4f}, {:.4f}, {:.4f}".format(
            macro_precision, macro_recall, macro_f1))

        for alpha in alpha_list:
            m = metrics_by_alpha[alpha]
            print("alpha={:.1f}, tail_size={}, CVaR={:.6f}, AC={:.4f}, PR={:.4f}, RC={:.4f}, F1={:.4f}".format(
                alpha, m['tail_size'], m['CVaR'], m['accuracy'],
                m['macro_precision'], m['macro_recall'], m['macro_f1']))

    # Save sample-level outputs only for explicit test evaluation calls.
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        per_sample = pd.DataFrame({
            'sample_index': np.arange(sample_count, dtype=np.int64),
            'loss': loss_list.numpy(),
            'gold_label': gold_all.numpy(),
            'pred_label': pred_all.numpy(),
            'correct': (pred_all == gold_all).numpy().astype(np.int8),
        })
        for alpha in alpha_list:
            per_sample[f'is_tail_alpha_{alpha:.1f}'] = tail_membership[alpha]
        per_sample.to_csv(os.path.join(save_path, 'per_sample_results.csv'), index=False, encoding='utf-8')

        for alpha in alpha_list:
            np.savetxt(
                os.path.join(save_path, f"tail_confusion_alpha_{alpha:.1f}.txt"),
                tail_confusions[alpha].numpy(), fmt='%d'
            )
            np.savetxt(
                os.path.join(save_path, f"tail_per_class_alpha_{alpha:.1f}.txt"),
                metrics_by_alpha[alpha]['per_class'], fmt='%.10f'
            )

        if dataset_name is None:
            raise ValueError("dataset_name must be provided when save_path is set.")
        _append_results_summary(args, dataset_name, metrics_by_alpha)

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(accuracy, correct, sample_count))
    return accuracy, confusion, CVaR_list, acc_list, rem, metrics_by_alpha

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--only_test", type=bool, default=False,
                        help="only test")
    parser.add_argument("--rm_lr", type=float,
                        help="Learning rate of the PERO surrogate model.")
    parser.add_argument("--gdro_tau", type=float, default=1.0,
                        help="Temperature for instance-level GroupDRO softmax weighting.")
    parser.add_argument("--tdro_rho", type=float, default=1e-3,
                        help="Tail-risk parameter for TDRO.")
    parser.add_argument("--tdro_lambda", type=float, default=1.0,
                        help="Softplus smoothing parameter for TDRO.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focusing parameter for Focal Loss.")
    parser.add_argument("--focal_alpha", type=float, default=1.0,
                        help="Alpha weight for Focal Loss.")
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    
    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(args.device)
    for p in model.embedding.parameters():
        p.requires_grad = False
    # Training phase.
    trainset = read_dataset(args, args.train_path)

    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)
    if args.method == 'PERO':
        riskmodel = SurrogateLossPredictor(args)
        
        
        
        riskmodel = riskmodel.to(args.device)
        if args.rm_lr == None:
            args.rm_lr = args.learning_rate
        optim = torch.optim.Adam(riskmodel.parameters(),lr=args.rm_lr)
        risk_lossfn = nn.MSELoss()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    total_corr = 0.0
    total_spearman_corr = 0.0
    total_prec_k = 0.0
    total_time = 0.0
    corr_list = []
    spearman_list = []
    precision_at_k_list = []
    time_list = []
    save_path = f"./{args.method}_app"
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(save_path+'/models',exist_ok=True)
    if args.only_test:
        if args.test_path is not None:
            print("Test set evaluation.")
            if torch.cuda.device_count() > 1:
                model.module.load_state_dict(torch.load(save_path+'/'+args.output_model_path,map_location='cuda:1'))
            else:
                model.load_state_dict(torch.load(save_path+'/'+args.output_model_path))
            with torch.no_grad():
                acc, confusion,CVaR_list,acc_list,rem,metrics_by_alpha = evaluate(args, read_dataset(args, args.test_path), True, save_path=save_path, dataset_name="app")

            r1 = np.array([acc]) 
            r2 = np.array(confusion)
            r3 = np.array(CVaR_list)
            r4 = np.array(acc_list)
            r5 = np.array(rem)
            np.savetxt(save_path+"/acc.txt", r1) 
            np.savetxt(save_path+"/confusion.txt", r2)
            np.savetxt(save_path+"/CVaR.txt", r3)
            np.savetxt(save_path+"/acc_list.txt", r4)
            np.savetxt(save_path+"/rem.txt", r5)
        else:
            raise ValueError('please input the test_path')
        return
    print("Start training.")
    
    
    total_steps = args.train_steps
    monitor = TrainingEfficiencyMonitor(
    experiment_name="exp1",
    method_name=args.method
    ).start()
    std = 1.
    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.LongTensor([example[0] for example in trainset])
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg = torch.LongTensor([example[2] for example in trainset])

        if args.soft_targets:
            soft_tgt = torch.FloatTensor([example[3] for example in trainset])
        else:
            soft_tgt = None
        model.train()

        
        

        
        batches_per_epoch_default = instances_num // batch_size + (1 if instances_num % batch_size > 0 else 0)

        
        ro_batch_size = int(batch_size/(1-args.CVaR_alpha))
        batches_per_epoch_ro = instances_num // ro_batch_size + (1 if instances_num % ro_batch_size > 0 else 0)

        
        if args.method in {'ERM', 'GroupDRO', 'TDRO', 'Focal'}:
            current_batches_per_epoch = batches_per_epoch_default
        else:
            current_batches_per_epoch = batches_per_epoch_ro

        steps_in_prev_epochs = (epoch - 1) * current_batches_per_epoch
        # ----------------------------------------------------

        if args.method == 'ERM':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                loss,_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    time_list.append(total_time / args.report_steps)
                    total_loss = 0.0
                    total_time = 0.0 
                    # ----------------------------------------------------

        elif args.method == 'Focal':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                loss,_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps
                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0


        elif args.method == 'MC-CVaR':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(int(batch_size/(1-args.CVaR_alpha)), src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                with torch.no_grad():
                    _,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch,inference=True)
                _,index = torch.topk(loss_.squeeze(),k=int((1-args.CVaR_alpha)*len(tgt_batch)))
                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)
                src_batch = src_batch[index]
                tgt_batch = tgt_batch[index]
                seg_batch = seg_batch[index]
                if soft_tgt_batch != None:
                    soft_tgt_batch = soft_tgt_batch[index]
                loss,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    time_list.append(total_time / args.report_steps)
                    total_loss = 0.0
                    total_time = 0.0 
                    # ----------------------------------------------------

        elif args.method == 'Random':
            candidate_batch_size = int(batch_size/(1-args.CVaR_alpha))
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(candidate_batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                random_k = min(batch_size, len(tgt_batch))
                index = torch.randperm(len(tgt_batch))[:random_k]
                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)
                index = index.to(args.device)
                src_batch = src_batch[index]
                tgt_batch = tgt_batch[index]
                seg_batch = seg_batch[index]
                if soft_tgt_batch != None:
                    soft_tgt_batch = soft_tgt_batch[index]
                loss,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps
                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0


        elif args.method == 'GroupDRO':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                model.zero_grad()

                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)

                _, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                weights = (loss_.detach() / args.gdro_tau).softmax(0)
                loss = torch.sum(weights * loss_)
                if torch.cuda.device_count() > 1:
                    
                    loss_ = loss_.view(-1)
                    weights = (loss_.detach() / args.gdro_tau).softmax(0)
                    loss = torch.sum(weights * loss_)
                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    time_list.append(total_time / args.report_steps)
                    total_loss = 0.0
                    total_time = 0.0 
                    # ----------------------------------------------------

        elif args.method == 'TDRO':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                model.zero_grad()

                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)

                _, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                if torch.cuda.device_count() > 1:
                    loss_ = loss_.view(-1)

                tdro_alpha = get_tdro_alpha(model)
                loss = tdro_softplus_loss(
                    per_sample_loss=loss_,
                    alpha=tdro_alpha,
                    rho=args.tdro_rho,
                    lam=args.tdro_lambda
                )

                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if args.clip:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time - start_time
                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]
                    current_alpha = get_tdro_alpha(model).detach().item()

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, TDRO alpha: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, current_alpha, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps

                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    with open(f"{save_path}/logs/{args.method}_alpha.txt", "a") as f4:
                        f4.write(f"{current_alpha}\n")
                    total_loss = 0.0
                    total_time = 0.0
                    # ----------------------------------------------------


        elif args.method == 'OHTM':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(int(batch_size/(1-args.CVaR_alpha)), src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                with torch.no_grad():
                    src_batch = src_batch.to(args.device)
                    seg_batch = seg_batch.to(args.device)
                    tgt_batch = tgt_batch.to(args.device)
                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch.to(args.device)

                    if torch.cuda.device_count() > 1:
                        emb = model.module.embedding(src_batch,seg_batch)
                    else:
                        emb = model.embedding(src_batch,seg_batch)

                while emb.dim() > 2:
                    emb = emb.mean(dim=1)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                n, d = emb.shape
                selected_indices = []
                mask = torch.ones(n, dtype=torch.bool, device=emb.device)
                residuals = emb.clone()
                
                squared_errors = (emb ** 2).sum(dim=1)  

                for _ in range(min(batch_size, n)):
                    curr_candidates = torch.where(mask, squared_errors, torch.tensor(-1.0, device=emb.device))
                    best_idx = torch.argmax(curr_candidates).item()

                    selected_indices.append(best_idx)
                    mask[best_idx] = False

                    e = residuals[best_idx] / torch.sqrt(squared_errors[best_idx] + 1e-9)
                    e = e.view(1, -1)

                    dots = torch.matmul(residuals, e.t()).squeeze()
                    residuals = residuals - dots.view(-1, 1) * e
                    squared_errors = torch.clamp(squared_errors - dots**2, min=0.0)

                index = selected_indices
                src_batch = src_batch[index]
                tgt_batch = tgt_batch[index]
                seg_batch = seg_batch[index]
                emb = emb[index]
                if soft_tgt_batch != None:
                    soft_tgt_batch = soft_tgt_batch[index]
                loss,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))
                    
                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps

                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0 



        elif args.method == 'PERO':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(int(batch_size/(1-args.CVaR_alpha)), src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                with torch.no_grad():
                    src_batch = src_batch.to(args.device)
                    seg_batch = seg_batch.to(args.device)
                    tgt_batch = tgt_batch.to(args.device)
                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch.to(args.device)

                    if torch.cuda.device_count() > 1:
                        emb = model.module.embedding(src_batch,seg_batch)
                    else:
                        emb = model.embedding(src_batch,seg_batch)

                risk = riskmodel(emb,tgt_batch,soft_tgt_batch)
                score = risk
                risk,index = torch.topk(score.squeeze(),k=int((1-args.CVaR_alpha)*len(tgt_batch)))
                src_batch = src_batch[index]
                tgt_batch = tgt_batch[index]
                seg_batch = seg_batch[index]
                emb = emb[index]
                if soft_tgt_batch != None:
                    soft_tgt_batch = soft_tgt_batch[index]
                loss,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                optim.zero_grad()
                riskloss = risk_lossfn(risk,loss_.detach())
                riskloss.backward()
                optim.step()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                pred_np = risk.detach().view(-1).cpu().numpy()
                true_np = loss_.detach().view(-1).cpu().numpy()
                pearson_corr = compute_pearson_corr(pred_np, true_np)["pearson_r"]
                spearman_corr = compute_spearman_corr(pred_np, true_np)["spearman_rho"]
                k_eff = max(1, int(0.5 * len(pred_np)))
                prec_k = precision_at_k(pred_np, true_np, k=k_eff)
                total_corr += pearson_corr
                total_spearman_corr += spearman_corr
                total_prec_k += prec_k
                if (i + 1) % args.report_steps == 0:
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    avg_pearson = total_corr / args.report_steps
                    avg_spearman = total_spearman_corr / args.report_steps
                    avg_prec_k = total_prec_k / args.report_steps
                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg Pearson: {:.3f}, Avg Spearman: {:.3f}, Avg Precision@K: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_pearson, avg_spearman, avg_prec_k, avg_time_per_step, estimated_time_remaining))
                    corr_list.append(avg_pearson)
                    spearman_list.append(avg_spearman)
                    precision_at_k_list.append(avg_prec_k)
                    time_list.append(total_time / args.report_steps)
                    total_loss = 0.0
                    total_corr = 0.0
                    total_spearman_corr = 0.0
                    total_prec_k = 0.0
                    total_time = 0.0
                    # ----------------------------------------------------
        monitor.record_checkpoint(f"epoch_{epoch}")
        corr_arr = np.array(corr_list)
        time_arr = np.array(time_list)
        np.savetxt(save_path+"/time.txt", time_arr)
        if args.method == 'PERO':
            np.savetxt(save_path+"/corr.txt", corr_arr)
            np.savetxt(save_path+"/spearman.txt", np.array(spearman_list))
            np.savetxt(save_path+"/precision_at_k.txt", np.array(precision_at_k_list))
        with torch.no_grad():
            result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, save_path+'/'+args.output_model_path)

    # Evaluation phase.
    monitor.stop()
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(save_path+'/'+args.output_model_path,map_location='cuda:1'))
        else:
            model.load_state_dict(torch.load(save_path+'/'+args.output_model_path))
        with torch.no_grad():
            acc, confusion,CVaR_list,acc_list,rem,metrics_by_alpha = evaluate(args, read_dataset(args, args.test_path), True, save_path=save_path, dataset_name="app")

        r1 = np.array([acc]) 
        r2 = np.array(confusion)
        r3 = np.array(CVaR_list)
        r4 = np.array(acc_list)
        r5 = np.array(rem)
        np.savetxt(save_path+"/acc.txt", r1) 
        np.savetxt(save_path+"/confusion.txt", r2)
        np.savetxt(save_path+"/CVaR.txt", r3)
        np.savetxt(save_path+"/acc_list.txt", r4)
        np.savetxt(save_path+"/rem.txt", r5)

if __name__ == "__main__":
    main()
