import sys
import os
import random
import copy
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from evaluate import load as load_metric
from torch.nn.functional import log_softmax
from transformers import AutoModelForSequenceClassification,AdamW
from peft import (
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
)
from arguments import parse_args, Args
from data_loader import load_data
from DDQN_agent import DDQNAgent
from Loss_estimation import fit_exp_and_predict
from torch.nn.functional import log_softmax
import os, ssl

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["HF_HUB_OFFLINE"] = "1"




reruns = 1
if "--reruns" in sys.argv:
    idx = sys.argv.index("--reruns")
    if idx + 1 < len(sys.argv):
        reruns = int(sys.argv[idx+1])
        sys.argv.pop(idx); sys.argv.pop(idx)
    else:
        raise ValueError("Provide an integer after --reruns")


args: Args = parse_args()
random.seed(0); np.random.seed(0)
torch.manual_seed(0); torch.cuda.manual_seed_all(0)

NUM_CLIENTS = args.num_clients
NUM_SPLITS  = NUM_CLIENTS + 1
DEVICE      = torch.cuda.current_device()
CHECKPOINT  = args.client_ckpt


DEVICE_IDX = torch.cuda.current_device()      # still an int, e.g. 0
MAP_LOC = f"cuda:{DEVICE_IDX}"

eps, eps_min, eps_dec = 1.0, 0.05, 0.995


AGENT_CKPT = "ddqn_agent.pth"
agent = DDQNAgent(num_clients=NUM_CLIENTS, lr=1e-3, gamma=0.9)
agent.policy_net.to(DEVICE)
agent.target_net.to(DEVICE)
if os.path.exists(AGENT_CKPT):
    ckpt = torch.load(AGENT_CKPT, map_location=MAP_LOC)
    agent.policy_net.load_state_dict(ckpt["policy"])
    agent.target_net.load_state_dict(ckpt["target"])
    agent.steps = ckpt.get("steps", 0)
    print(f"🔄 Loaded agent from {AGENT_CKPT}")

history = {"val_losses": [], "val_accuracies": []}


def smoothed_loss(logits, targets, smoothing=0.1):
    K = logits.size(-1)
    logp = log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.full_like(logp, smoothing/(K-1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1-smoothing)
    return torch.mean((-true_dist * logp).sum(dim=-1))

def train_epochs(model, loader, epochs, lr):
    opt = AdamW(model.parameters(), lr=lr, no_deprecation_warning=True)
    model.to(DEVICE)
    losses = []
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            opt.zero_grad()
            logits = model(**batch).logits
            loss = smoothed_loss(logits, batch["labels"])
            loss.backward(); opt.step()
            total += loss.item()*batch["labels"].size(0)
            count += batch["labels"].size(0)
        losses.append(total/count)
    return losses

def test(net, loader,DEVICE):
    metric = load_metric("accuracy")
    net.eval()
    net.to(DEVICE)
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            out = net(**batch)
            total += out.loss.item()*len(batch["labels"])
            metric.add_batch(predictions=torch.argmax(out.logits, dim=-1),
                             references=batch["labels"])
            count += len(batch["labels"])
    return total/count, metric.compute()["accuracy"]

def pad_to_shape(arr, shape):
    padded = np.zeros(shape, dtype=arr.dtype)
    slices = tuple(slice(0, min(d,s)) for d,s in zip(arr.shape, shape))
    padded[slices] = arr[slices]
    return padded
def adapt_aggregated_state(aggregated_state, local_state):
    adapted_state = {}
    for key, local_param in local_state.items():
        if key in aggregated_state:
            agg_param = aggregated_state[key]
            if "lora_A" in key:
                expected_rank = local_param.shape[0]
                if agg_param.shape[0] > expected_rank:
                    agg_param = agg_param[:expected_rank, ...]
                elif agg_param.shape[0] < expected_rank:
                    pad_size = expected_rank - agg_param.shape[0]
                    pad = torch.zeros(pad_size, *agg_param.shape[1:], device=agg_param.device)
                    agg_param = torch.cat([agg_param, pad], dim=0)
            elif "lora_B" in key:
                expected_rank = local_param.shape[1]
                if agg_param.shape[1] > expected_rank:
                    agg_param = agg_param[:, :expected_rank]
                elif agg_param.shape[1] < expected_rank:
                    pad_size = expected_rank - agg_param.shape[1]
                    pad = torch.zeros(*agg_param.shape[:1], pad_size, device=agg_param.device)
                    agg_param = torch.cat([agg_param, pad], dim=1)
            adapted_state[key] = agg_param
        else:
            adapted_state[key] = local_param
    return adapted_state

def custom_fedavg_aggregate(client_updates, global_keys):
    client_states = []
    for params_list, mask, num_examples in client_updates:
        state = params_list  # now always a dict
        client_states.append((state, mask, num_examples))

    total_weight = sum(n for _, _, n in client_states)
    aggregated = {}

    for key in global_keys:
        updates, local_weights = [], []
        for state, mask, num_examples in client_states:
            if mask.get(key, False):
                # move to CPU and detach before numpy conversion
                tensor = state[key].detach().cpu()
                arr = tensor.numpy().astype(np.float32)
                updates.append(arr)
                local_weights.append(num_examples)

        if not updates:
            continue

        if "lora_A" in key:
            scaled = [u * (w / total_weight) for u, w in zip(updates, local_weights)]
            final = np.concatenate(scaled, axis=0)

        elif "lora_B" in key:
            final = np.concatenate(updates, axis=1)

        else:
            # average (with padding if needed)
            first_shape = updates[0].shape
            if all(u.shape == first_shape for u in updates):
                final = np.average(updates, axis=0, weights=local_weights)
            else:
                max_shape = list(first_shape)
                for u in updates[1:]:
                    max_shape = [max(m, s) for m, s in zip(max_shape, u.shape)]
                padded = [pad_to_shape(u, tuple(max_shape)) for u in updates]
                final = np.average(padded, axis=0, weights=local_weights)

        aggregated[key] = final.astype(np.float32)

    return [aggregated.get(k, np.zeros(0,)) for k in global_keys]

def convert_state_to_list(client_state):
    sorted_keys = sorted(client_state.keys())
    params_list = [client_state[k] for k in sorted_keys]
    mask = {str(i): True for i in range(len(params_list))}
    return params_list, mask, sorted_keys




def sequential_fedavg():
    global eps

    base = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2,local_files_only=True)
    peft_cfg = LoraConfig(
        task_type="SEQ_CLS", inference_mode=False,
        r=args.lora_r, lora_alpha=args.lora_r,
        target_modules=["q_lin","v_lin"], lora_dropout=0.1
    )
    global_model = get_peft_model(base, peft_cfg)
    global_model.to(DEVICE)

    g0 = get_peft_model_state_dict(global_model)

    ordering = sorted(g0.keys())
    global_keys = ordering

    prev_acc = 0.0

    for rnd in range(1, args.num_rounds+1):
        print(f"\n--- Round {rnd}/{args.num_rounds} ---")

        state = []
        for cid in range(1, NUM_CLIENTS+1):
            print(f" [Probe] client {cid}")
            trn,_ = load_data(args.data_path, args.data_name, cid, NUM_SPLITS, CHECKPOINT)
            probe = copy.deepcopy(global_model)
            losses3 = train_epochs(probe, trn, epochs=3, lr=args.client_lr)
            est, _ = fit_exp_and_predict(losses3, fit_pts=[1,2,3], pred_pts=[args.client_epochs])
            state.append(est[0])
        state_vec = torch.tensor(state, dtype=torch.float32, device=DEVICE)  # shape [N]
        k = max(1, NUM_CLIENTS // 2)
        sel = agent.select_action(state_vec.unsqueeze(0), eps, k)
        sel_list = sel.cpu().tolist()  # Python list of ints
        print(f" [Select] {sel_list}")

        updates = []
        for cid in sel_list:  # cid is now an int
            print(f" [Train] client {cid}")
            trn, tst = load_data(args.data_path,
                                 args.data_name,
                                 cid,  # plain int
                                 NUM_SPLITS,
                                 CHECKPOINT)

            client = copy.deepcopy(global_model)
            losses = train_epochs(client, trn, epochs=args.client_epochs, lr=args.client_lr)
            params = get_peft_model_state_dict(client)
            updates.append((params, {k:True for k in global_keys}, len(trn.dataset)))

        print(" [Aggregate]")
        agg_list = custom_fedavg_aggregate(updates, global_keys)
        new_params = { ordering[i]: agg_list[i] for i in range(len(ordering)) }

        g_state = get_peft_model_state_dict(global_model)
        merged = {}
        for k, old in g_state.items():
            if k in new_params:
                merged[k] = torch.tensor(new_params[k], device=DEVICE)
            else:
                merged[k] = old

        set_peft_model_state_dict(global_model, adapt_aggregated_state(merged, g_state))

        val_loss, val_acc = test(global_model, tst,DEVICE=DEVICE)
        print(f" [Eval] loss={val_loss:.4f}, acc={val_acc:.4f}")
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        reward = val_acc - prev_acc
        prev_acc = val_acc
        print(f" [Reward] {reward:.4f}")
        k = max(1, NUM_CLIENTS // 2)
        sel = agent.select_action(state_vec, eps, k)

        print(f" [Select] {sel.tolist()}")



        next_state_t = state_vec.clone()

        agent.push_transition(state_vec, sel, reward, next_state_t)

        agent.update(batch_size=32)

        torch.save({
            "policy": agent.policy_net.state_dict(),
            "target": agent.target_net.state_dict(),
            "steps":  agent.steps
        }, AGENT_CKPT)
        print(" [Checkpoint] agent saved")

        eps = max(eps_min, eps * eps_dec)
        print(f" [Eps] {eps:.3f}")

    with open("fedavg_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("📑 Saved history to fedavg_history.json")




if __name__ == "__main__":
    for run_id in range(1, reruns+1):
        print(f"\n=== Run {run_id}/{reruns} ===")
        sequential_fedavg()
