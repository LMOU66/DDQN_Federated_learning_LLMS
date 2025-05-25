import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import curve_fit
from torch.nn.functional import log_softmax
from transformers import AutoModelForSequenceClassification,AdamW
from peft import get_peft_model, LoraConfig
from arguments import parse_args, Args
from data_loader import load_data
import random

SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


DEVICE = torch.cuda.current_device()



def smoothed_loss(logits, targets, smoothing=0.1):
    K = logits.size(-1)
    logp = log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.full_like(logp, smoothing/(K-1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1-smoothing)
    return torch.mean((-true_dist * logp).sum(dim=-1))

def train_epochs(model, loader, epochs, lr):
    opt = AdamW(model.parameters(), lr=lr, no_deprecation_warning=True)
    losses = []
    model.to(DEVICE)
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            opt.zero_grad()
            logits = model(**batch).logits
            loss = smoothed_loss(logits, batch["labels"])
            loss.backward()
            opt.step()
            total += loss.item() * batch["labels"].size(0)
            count += batch["labels"].size(0)
        losses.append(total/count)
    return losses

def exp_offset(t, a, b, c):
    return a * np.exp(-b * t) + c

def fit_exp_and_predict(losses, fit_pts, pred_pts):
    x = np.array(fit_pts)
    y = np.array([losses[t-1] for t in fit_pts])
    p0 = [y[0]-y[-1], 0.5, y[-1]]
    try:
        (a,b,c), _ = curve_fit(exp_offset, x, y, p0=p0, maxfev=5000)
    except Exception:
        a,b,c = p0
    preds = exp_offset(np.array(pred_pts), a, b, c)
    return preds, (a,b,c)

def main():
    args: Args = parse_args()

    num_clients = args.num_clients
    lr          = args.client_lr
    ckpt        = args.client_ckpt
    lora_rank   = args.lora_r
    real_epochs = args.client_epochs

    probe_epochs = [1,2,3]
    all_real = {}
    all_est  = {}


    backbone = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
    peft_cfg = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q_lin","v_lin"],
        lora_dropout=0.1,
        use_rslora=False,
    )
    base_model = get_peft_model(backbone, peft_cfg)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(num_clients)]


    for cid in range(1, num_clients+1):
        print(f"Client {cid}", end=" ")

        trainloader, _ = load_data(
            args.data_path, args.data_name,
            cid, num_clients+1, ckpt
        )

        real_model   = copy.deepcopy(base_model)
        real_curve   = train_epochs(real_model, trainloader, real_epochs, lr)

        probe_model  = copy.deepcopy(base_model)
        probe_curve  = train_epochs(probe_model, trainloader, len(probe_epochs), lr)

        if len(probe_curve) < max(probe_epochs):
            probe_curve += [probe_curve[-1]] * (max(probe_epochs) - len(probe_curve))

        predict_pts = list(range(1, real_epochs+1))
        est_curve, params = fit_exp_and_predict(
            probe_curve, fit_pts=probe_epochs, pred_pts=predict_pts
        )
        print(f"β={tuple(round(x,3) for x in params)}")

        all_real[cid] = real_curve
        all_est[cid]  = est_curve


    plt.figure(figsize=(10,6))
    for cid in range(1, num_clients+1):
        clr = colors[(cid-1) % len(colors)]
        x = list(range(1, real_epochs+1))
        plt.plot(x, all_real[cid],   '-o', color=clr, label=f"C{cid} Real")
        plt.plot(x, all_est[cid],    '--',  color=clr, label=f"C{cid} Est")

    plt.title(f"Real vs Estimated Loss Curves (LoRA rank={lora_rank})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small", ncol=1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
