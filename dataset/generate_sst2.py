import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
import sys
import random
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import DistilBertTokenizer
from utils.dataset_utils import check, separate_data, split_data, save_file, separate_data_sst
import shutil

random.seed(1)
np.random.seed(1)
num_clients = 100
num_classes = 2
dir_path = "SST2/"


def generate_sst2(dir_path, num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if niid == True:
        dist = 'noniid'
        if partition == 'pat':
            dist = dist + '_pat'
        elif partition == 'dir':
            dist = dist + '_dir'
    else:
        dist = 'iid'

    path = dir_path + dist

    if not os.path.exists(path):
        os.makedirs(path)

    config_path = path + "/config.json"
    train_path = path + "/train/"
    test_path = path + "/test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # 方法1：指定缓存目录并清除旧缓存
    cache_dir = "./cache/sst2"
    if os.path.exists(cache_dir):
        print("Clearing old cache...")
        shutil.rmtree(cache_dir)

    print("Loading SST-2 dataset...")
    try:
        # 指定缓存目录加载数据集
        dataset = load_dataset("glue", "sst2", download_mode='force_redownload', cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative method...")
        # 方法2：使用 trust_remote_code
        dataset = load_dataset("glue", "sst2", trust_remote_code=True, cache_dir=cache_dir)

    # 初始化 tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # 合并训练集和验证集
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']
    combined_dataset = concatenate_datasets([train_dataset, valid_dataset])

    # 提取文本和标签
    all_texts = list(combined_dataset['sentence'])
    all_labels = list(combined_dataset['label'])

    print(f"Total samples: {len(all_texts)}")
    print(f"Tokenizing texts...")

    # Tokenize所有文本
    encoded = tokenizer(
        all_texts,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # 准备数据格式
    dataset_input_ids = encoded['input_ids'].numpy()
    dataset_attention_mask = encoded['attention_mask'].numpy()
    dataset_label = np.array(all_labels)

    # 将 input_ids 和 attention_mask 合并
    dataset_content = np.concatenate([dataset_input_ids, dataset_attention_mask], axis=1)

    print(f"Dataset content shape: {dataset_content.shape}")
    print(f"Dataset label shape: {dataset_label.shape}")
    print(f"Label distribution: {np.bincount(dataset_label)}")

    # 使用 separate_data 划分数据
    X, y, statistic = separate_data_sst(
        (dataset_content, dataset_label),
        num_clients,
        num_classes,
        niid,
        balance,
        partition,
        class_per_client=1 if niid else num_classes,
        alpha=0.5,  # 添加：控制非IID程度，越小越不均匀
        least_samples=10  # 添加：每个客户端最少样本数
    )

    # 分割训练集和测试集
    train_data, test_data = split_data(X, y)

    # 保存文件
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition)

    print("SST-2 dataset generation completed!")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_sst2(dir_path, num_clients, num_classes, niid, balance, partition)
