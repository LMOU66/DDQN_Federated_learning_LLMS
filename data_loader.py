from datasets import disable_progress_bar
disable_progress_bar()

import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from datasets import logging as ds_logging
ds_logging.set_verbosity_error()

PARTITION_CACHE = {}


def _load_data(train_raw_dataset, test_raw_dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_datasets = train_raw_dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["idx", "sentence"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    test_tokenized_datasets = test_raw_dataset.map(tokenize_function, batched=True)
    test_tokenized_datasets = test_tokenized_datasets.remove_columns(["idx", "sentence"])
    test_tokenized_datasets = test_tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets, shuffle=True, batch_size=32, collate_fn=data_collator
    )

    testloader = DataLoader(
        test_tokenized_datasets, batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def partition_text(alpha, num_clients, labels, num_classes=2, min_samples=100):
    min_size = 0
    N = len(labels)
    net_dataidx_map = {}

    while min_size < min_samples:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet([alpha] * num_clients)

            proportions = np.array([
                p * (len(batch) < (N / num_clients))
                for p, batch in zip(proportions, idx_batch)
            ])
            proportions = proportions / proportions.sum()

            cum_indices = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            split_indices = np.split(idx_k, cum_indices)
            for i in range(num_clients):
                idx_batch[i].extend(split_indices[i].tolist())

        sizes = [len(b) for b in idx_batch]
        min_size = min(sizes)

    for j in range(num_clients):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def load_data(
        path,
        name,
        rank,
        num_splits,
        tokenizer_ckpt,
        teacher_data_pct=0.3,
        num_classes=2,
        min_samples=100):
    alpha = teacher_data_pct if teacher_data_pct > 0 else 0.5
    partition_key = (path, name, num_splits, alpha, min_samples)

    if partition_key not in PARTITION_CACHE:
        full_dataset = load_dataset(path, name, split="train")
        labels = np.array(full_dataset["label"])

        net_dataidx_map = partition_text(
            alpha=alpha,
            num_clients=num_splits,
            labels=labels,
            num_classes=num_classes,
            min_samples=min_samples
        )
        PARTITION_CACHE[partition_key] = {
            "dataset": full_dataset,
            "map": net_dataidx_map
        }
    else:
        cached = PARTITION_CACHE[partition_key]
        full_dataset = cached["dataset"]
        net_dataidx_map = cached["map"]

    test_raw_dataset = load_dataset(path, name, split="validation")

    client_indices = net_dataidx_map[rank]
    client_dataset = full_dataset.select(client_indices)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ckpt,local_files_only=True)
    trainloader, testloader = _load_data(client_dataset, test_raw_dataset, tokenizer)
    return trainloader, testloader
