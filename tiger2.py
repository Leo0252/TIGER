import argparse
import numpy as np
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration
from torch.optim import Adam
from tqdm import tqdm
import logging
import os
import pandas as pd
import pyarrow.parquet as pq

import collections
import os
from tqdm import tqdm
import polars as pl

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name.')
    parser.add_argument('--init_way', type=str, required=True, help='Init way of embeddings.')

    return parser.parse_args()


def get_paths(dataset_name, init_way):
    base_path = f'./data/{dataset_name}/'
    ckpt_path = f'./ckpt_tiger/{dataset_name}/{init_way}/'
    
    # 定义所有路径
    paths = {
        'data_path': base_path,
        'item_codes_path': base_path + f'codes_{init_way}.parquet',
        'train_path': base_path + 'train.parquet',
        'test_path': base_path + 'test.parquet',
        'valid_path': base_path + 'valid.parquet',
        'model_path': ckpt_path + 'model.pth',
        'model_best_path': ckpt_path + 'model_best.pth'
    }
    
    # 检查并创建所有路径的父目录
    for path in paths.values():
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    return paths




# 解析参数 
args = parse_args() 
paths = get_paths(args.dataset_name, args.init_way) 
# 设置日志记录 
logging.basicConfig(filename=os.path.join(paths['data_path'], f'tiger_log/output_{args.init_way}.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 
logger = logging.getLogger()
# 文件路径配置 
item_codes_path = paths['item_codes_path'] 
train_path = paths['train_path'] 
test_path = paths['test_path'] 
valid_path = paths['valid_path'] 
model_path = paths['model_path'] 
model_best_path = paths['model_best_path']


# 超参数设置
EPOCHS = 1000
BATCH_SIZE = 256
EMBEDDING_DIM = 128
NUM_HEADS = 6
MLP_DIM = 1024
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROPOUT = 0.1
max_length = 5
min_seq = 5
max_seq = 20
num_beams = 30
vocab_size = 256 * 4 + 1
PAD_VALUE = 256 * 4


# 数据处理函数
def get_codes_sequence(item_sequence):
    # codes = [item_codes_map[item_id]['code'] for item_id in item_sequence]
    codes = [item_codes_map[item_id] for item_id in item_sequence]
    codes = [item for sublist in codes for item in sublist]  # 将嵌套列表展开为一维列表
    return np.array(codes)

def get_codes_item(item):
    # return np.array(item_codes_map[item]['code'])
    return np.array(item_codes_map[item])

def pad_sequence(sequence, max_length, PAD_VALUE):
    return list(sequence) + [PAD_VALUE] * (max_length - len(sequence))

def create_attention_matrix(sequences, padding_value):
    return np.array([[1 if item != padding_value else 0 for item in seq] for seq in sequences])

def get_batches(train, batchsize, pad_token_id):
    input_sequences = np.array([np.array(seq) for seq in train['input_sequences'].to_numpy()])
    target_codes = np.array([np.array(seq) for seq in train['target_codes'].to_numpy()])
    num_batches = len(train) // batchsize
    batches = []
    for batch_idx in range(num_batches):
        batch_input_sequences = input_sequences[batch_idx * batchsize: (batch_idx + 1) * batchsize]
        batch_target_codes = target_codes[batch_idx * batchsize: (batch_idx + 1) * batchsize]
        batch_attention_matrix = create_attention_matrix(batch_input_sequences, pad_token_id)
        batches.append({
            'input_sequences': batch_input_sequences,
            'attention_matrix': batch_attention_matrix,
            'target_matrix': batch_target_codes
        })
    return batches


def truncate_train(row, max_seq):
    item_sequence = row['item_sequence']
    if len(item_sequence) > max_seq-3:  # 训练集超过max_seq-3的行直接删除
        return None
    return row

def truncate_valid(row, max_seq):
    item_sequence = row['item_sequence']
    if len(item_sequence) > max_seq-2:  # 验证集超过max_seq-2，将第max_seq-2个作为target
        row['target'] = item_sequence[max_seq-2]
        row['item_sequence'] = item_sequence[:max_seq-2]
    return row

def truncate_test(row, max_seq):
    item_sequence = row['item_sequence']
    if len(item_sequence) > max_seq-1:  # 测试集超过max_seq-1，将第max_seq-1个作为target
        row['target'] = item_sequence[max_seq-1]
        row['item_sequence'] = item_sequence[:max_seq-1]
    return row

def process_data(train, item_codes, batchsize, pad_token_id, max_seq, split_fn):
    train = train.apply(split_fn, axis=1, max_seq=max_seq)
    train.dropna(inplace=True)  # 删除包含 None 的行
    train['item_sequence_codes'] = train['item_sequence'].apply(get_codes_sequence)
    train['target_codes'] = train['target'].apply(get_codes_item)
    max_length = max(train['item_sequence_codes'].apply(len))
    train['input_sequences'] = train['item_sequence_codes'].apply(lambda x: pad_sequence(x, max_length, pad_token_id))
    return get_batches(train, batchsize, pad_token_id)




def get_codes_id(item_codes):
    codes_to_id = {}
    for _, row in item_codes.iterrows():
        code = tuple(row['code'])
        codes_to_id[code] = row['key']
    return codes_to_id

# 读取用户交互序列
train = pq.read_table(train_path).to_pandas()
test = pq.read_table(test_path).to_pandas()
valid = pq.read_table(valid_path).to_pandas()

item_codes = pq.read_table(item_codes_path).to_pandas()
# item_codes['code'] = item_codes.apply(lambda row: [row[f'code_{i}'] for i in range(4)], axis=1)
codes_id = get_codes_id(item_codes)
item_codes_map = item_codes.set_index('key')['code'].to_dict()
# print(item_codes_map)

# 使用特定的截断函数处理不同的数据集
train_batches = process_data(train, item_codes, BATCH_SIZE, PAD_VALUE, max_seq, truncate_train)
valid_batches = process_data(valid, item_codes, BATCH_SIZE, PAD_VALUE, max_seq, truncate_valid)
test_batches = process_data(test, item_codes, BATCH_SIZE, PAD_VALUE, max_seq, truncate_test)


import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration

# 推荐模型
class RecommendationModel(nn.Module):
    def __init__(self, vocab_size):
        super(RecommendationModel, self).__init__()
        config = T5Config(
            d_model=EMBEDDING_DIM,
            d_ff=MLP_DIM,
            num_layers=NUM_ENCODER_LAYERS,
            num_heads=NUM_HEADS,
            dropout_rate=DROPOUT,
            pad_token_id=PAD_VALUE,
            decoder_start_token_id=PAD_VALUE,
            vocab_size=vocab_size
        )
        self.model = T5ForConditionalGeneration(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

    def generate(self, input_ids, attention_mask=None, max_length=5, num_beams=3, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams, num_return_sequences=num_beams, **kwargs)


# 评价指标计算
def calculate_recall_at_k(recommendations, ground_truth, k):
    recall = 0
    for user_recommendations, user_ground_truth in zip(recommendations, ground_truth):
        hits = len(set(user_recommendations[:k]) & set(user_ground_truth))
        recall += hits
    return recall / len(recommendations)

def calculate_ndcg_at_k(recommendations, ground_truth, k):
    ndcg = 0
    for user_recommendations, user_ground_truth in zip(recommendations, ground_truth):
        actual_relevance = [1 if item in user_ground_truth else 0 for item in user_recommendations[:k]]
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(actual_relevance))
        ideal_relevance = [1] * min(k, len(user_ground_truth))
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))
        ndcg += dcg / idcg if idcg > 0 else 0
    return ndcg / len(recommendations)


def train_model(model, train_batches, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_batches):
        input_sequences = torch.tensor(batch['input_sequences'], dtype=torch.long).to(device)
        attention_matrix = torch.tensor(batch['attention_matrix'], dtype=torch.long).to(device)
        target_matrix = torch.tensor(batch['target_matrix'], dtype=torch.long).to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_sequences, attention_mask=attention_matrix, labels=target_matrix)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # # 日志记录每个 batch 的损失
        # logging.info(f"Batch loss: {loss.item()}")

    average_loss = total_loss / len(train_batches)
    logging.info(f"Average training loss: {average_loss}")
    return average_loss

def get_id(recommendations, code_to_item):
    result = []
    """设置这里L为code的长"""
    L=4
    # print(f"recommendation:{recommendations.shape}")
    for seq in recommendations:
        item_id = []
        for i in range(0, len(seq), L):
            code_pair = tuple(seq[i:i+L])
            item = code_to_item.get(code_pair)
            if item:
                item_id.append(item)
        result.append(item_id)
    return result

def remove_input_item(recommendations, input_sequences):
    batch_size = len(recommendations)
    for i in range(batch_size):
        input_item = set(input_sequences[i])
        recommendations[i] = [item for item in recommendations[i] if item not in input_item]
    return recommendations

def validate_model(model, val_batches, max_length, num_beams, device):
    model.eval()
    all_recommendations = []
    all_ground_truth = []
    with torch.no_grad():
        for batch in tqdm(val_batches):
            # 转换为张量并移动到设备上
            input_sequences = torch.tensor(batch['input_sequences'], dtype=torch.long).to(device)
            attention_matrix = torch.tensor(batch['attention_matrix'], dtype=torch.long).to(device)

            # 生成输出
            outputs = model.generate(input_ids=input_sequences, attention_mask=attention_matrix, max_length=max_length, num_beams=num_beams)

            # 获取推荐结果
            recommendations = outputs[:, 1:].cpu().tolist()  # 去除填充值
            recommendations = [seq for sublist in recommendations for seq in sublist]  # 平展列表
            recommendations = np.array(recommendations).reshape(len(batch['input_sequences']), num_beams * (max_length - 1))
            recommendations = get_id(recommendations, codes_id)
            
            # 获取输入序列的ID
            input_sequences = get_id(input_sequences.cpu().tolist(), codes_id)
            
            # 移除输入项目
            recommendations = remove_input_item(recommendations, input_sequences)

            all_recommendations.extend(recommendations)

            # 获取实际结果
            ground_truth = batch['target_matrix']
            ground_truth = get_id(ground_truth, codes_id)
            all_ground_truth.extend(ground_truth)

            # # 日志记录每个 batch 的推荐结果
            # logging.info(f"Batch recommendations: {recommendations}")

    return all_recommendations, all_ground_truth



def save_model(model, optimizer, epoch, filename='pth'):
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, filename)
    print(f"Model saved to {filename}")
    logging.info(f"Model saved to {filename}")

def load_model(model, optimizer, filename='pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Model loaded from {filename}, resuming from epoch {start_epoch}")
    logging.info(f"Model loaded from {filename}, resuming from epoch {start_epoch}")
    return start_epoch


# 主训练循环
def main_training_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = RecommendationModel(vocab_size).to(device)
    optimizer = Adam(model.parameters(), lr=3e-5)
    start_epoch=0
    logger.info('----------------------------------------------------------------------------------------------------------------------------------------------')
    recall_5_best = 0
    if os.path.exists(model_best_path):
        start_epoch=load_model(model, optimizer, model_best_path)

        recommendations, ground_truth = validate_model(model, valid_batches, max_length, num_beams, device)
        print(recommendations[0])
        print(ground_truth[0])
        recall_5 = calculate_recall_at_k(recommendations, ground_truth, 5)
        ndcg_5 = calculate_ndcg_at_k(recommendations, ground_truth, 5)
        print(f"验证集Recall@5: {recall_5:.4f}, NDCG@5: {ndcg_5:.4f}")
        logger.info(f"验证集Recall@5: {recall_5:.4f}, NDCG@5: {ndcg_5:.4f}")
        recall_10 = calculate_recall_at_k(recommendations, ground_truth, 10)
        ndcg_10 = calculate_ndcg_at_k(recommendations, ground_truth, 10)
        print(f"验证集Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}")
        logger.info(f"验证集Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}")
        recall_5_best = recall_5

    if os.path.exists(model_path):
        start_epoch=load_model(model, optimizer, model_path)
    for epoch in range(start_epoch, EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        logger.info(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_model(model, train_batches, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Loss: {train_loss:.4f}")

        if (epoch+1)%10==0 or epoch==0:
            recommendations, ground_truth = validate_model(model, valid_batches, max_length, num_beams, device)
            print(recommendations[0])
            print(ground_truth[0])
            recall_5 = calculate_recall_at_k(recommendations, ground_truth, 5)
            ndcg_5 = calculate_ndcg_at_k(recommendations, ground_truth, 5)
            print(f"验证集Recall@5: {recall_5:.4f}, NDCG@5: {ndcg_5:.4f}")
            logger.info(f"验证集Recall@5: {recall_5:.4f}, NDCG@5: {ndcg_5:.4f}")
            recall_10 = calculate_recall_at_k(recommendations, ground_truth, 10)
            ndcg_10 = calculate_ndcg_at_k(recommendations, ground_truth, 10)
            print(f"验证集Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}")
            logger.info(f"验证集Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}")
            save_model(model,optimizer,epoch,model_path)
            if recall_5 > recall_5_best:
                recall_5_best = recall_5
                save_model(model,optimizer,epoch,model_best_path)
                recommendations, ground_truth = validate_model(model, test_batches, max_length, num_beams, device)
                print(recommendations[0])
                print(ground_truth[0])
                #训练集上的评价指标 
                recall_5 = calculate_recall_at_k(recommendations, ground_truth, 5)
                ndcg_5 = calculate_ndcg_at_k(recommendations, ground_truth, 5)
                print(f"测试集Recall@5: {recall_5:.4f}, NDCG@5: {ndcg_5:.4f}")
                logger.info(f"测试集Recall@5: {recall_5:.4f}, NDCG@5: {ndcg_5:.4f}")

                recall_10 = calculate_recall_at_k(recommendations, ground_truth, 10)
                ndcg_10 = calculate_ndcg_at_k(recommendations, ground_truth, 10)
                print(f"测试集Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}")
                logger.info(f"测试集Recall@10: {recall_10:.4f}, NDCG@10: {ndcg_10:.4f}")

            




# 开始训练
main_training_loop()