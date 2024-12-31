# %% [markdown]
# torch ddp
# 启动命令：torchrun --nproc_per_node=2 分布式3torch_ddp.py

# %% [markdown]
# ## Step1 导入相关包

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.distributed as dist 

dist.init_process_group(backend="nccl")  # 初始化进程组，指定通信后端NCCL
# %% [markdown]
# ## Step2 加载数据

# %%
import pandas as pd

data = pd.read_csv("ChnSentiCorp_htl_all.csv")
data

# %%
data = data.dropna()
data

# %% [markdown]
# ## Step3 创建Dataset

# %%
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)

# %%
dataset = MyDataset()

# %% [markdown]
# ## Step4 划分数据集

# %%
from torch.utils.data import random_split
import torch

# 确保在不同进程中数据划分相同，防止指标虚高——验证：两个进程打印的5条数据相同
trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))  
len(trainset), len(validset)

# %%
for i in range(5):
    print(trainset[i])

# %% [markdown]
# ## Step5 创建Dataloader

# %%
import torch

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs

# %%
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 分布式的数据采样器,避免不同进程之间读取到重复的数据
trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, sampler=DistributedSampler(trainset))
validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, sampler=DistributedSampler(validset))

# %%
next(enumerate(validloader))[1]

# %% [markdown]
# ## Step6 创建模型及优化器

# %%
from torch.optim import Adam 
from torch.nn.parallel import DistributedDataParallel as DDP

model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")

if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))  # 决定该进程运行在哪块GPU上

model = DDP(model)

# %%
optimizer = Adam(model.parameters(), lr=2e-5)

# %% [markdown]
# ## Step7 训练与验证

# %%
def print_rank_0(args):
    if os.environ["RANK"] == '0':  # 全局rank
        print(args)

def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"]))  for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset) 

def train(epoch=3, log_step=100):
    global_step = 0
    for ep in range(epoch):
        model.train()
        trainloader.sampler.set_epoch(ep)  # 确保每个 epoch 在每个进程中进行不同的随机化操作
        for batch in trainloader:
            if torch.cuda.is_available():
                batch = {k: v.to(int(os.environ["LOCAL_RANK"]))  for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)  # 通信
                print_rank_0(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
            global_step += 1
        acc = evaluate() 
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)  # 通信
        print_rank_0(f"ep: {ep}, acc: {acc}")

# %% [markdown]
# ## Step8 模型训练

# %%
train()


