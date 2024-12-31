"""
batchsize 32：
普通accelerate：显存2.983G，时间21.29s
混合精度：显存2.665G，时间19.03s

batchsize 64：（激活值占比越大，混合精度越能节省显存）
普通accelerate：显存3.629G，时间17.11s
混合精度：显存3.171G，时间11.53s
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import time
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.distributed as dist 
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam 
from accelerate import Accelerator  # 第一行改动

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)


def prepare_dataloader():
    dataset = MyDataset()

    # 确保在不同进程中数据划分相同，防止指标虚高——验证：两个进程打印的5条数据相同
    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42)) 

    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    trainloader = DataLoader(trainset, batch_size=32, collate_fn=collate_func, shuffle=True)
    validloader = DataLoader(validset, batch_size=64, collate_fn=collate_func, shuffle=False)

    return trainloader, validloader


def prepare_model_and_optimizer():
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    optimizer = Adam(model.parameters(), lr=2e-5)
    return model, optimizer


def print_rank_0(args):
    if os.environ["RANK"] == '0':  # 全局rank
        print(args)


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    '''
    1、model.eval()仅作用于dropout层和batchnorm层;
    2、torch.no_grad()是在 pytorch 原有的机制上禁用了梯度的计算，底层还是有梯度计算的框架(autograd)，
    属于可以算，但是逻辑上不进行计算;
    3、with torch.inference_mode()完全脱离了 autograd 系统，而且把 View Tracking（视图追踪）和Version
    Counter Bumps（版本计数器更新）同样砍掉了。从底层上就完全放弃了梯度计算的逻辑。
    '''
    # with torch.inference_mode():
    with torch.no_grad():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)  # len(validloader.dataset)是sample数

def train(model, optimizer, trainloader, validloader, accelerator: Accelerator, resume, epoch=3, log_step=10):
    global_step = 0
    start_time = time.time()

    resume_epoch = 0
    resume_step = 0

    # 恢复检查点
    if resume:
        accelerator.load_state(resume)  # resume是个路径
        steps_per_epoch = math.ceil(len(trainloader) / accelerator.gradient_accumulation_steps)  # len(trainloader)是batch数
        resume_step = int(resume.split("step_")[-1])
        global_step = resume_step
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch
        accelerator.print(f"resume from: {resume_epoch} epoch {resume_step} step")

    for ep in range(resume_epoch, epoch):
        model.train()

        if resume and ep == resume_epoch and resume_step != 0:
            activate_dataloader = accelerator.skip_first_batches(trainloader, resume_step * accelerator.gradient_accumulation_steps)
        else:
            activate_dataloader = trainloader

        for batch in activate_dataloader:
            with accelerator.accumulate(model):  # 训练过程中，加入上下文
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss 
                accelerator.backward(loss)  # 第四行改动
                optimizer.step()

                if accelerator.sync_gradients:  # 梯度同步标志
                    global_step += 1

                    if global_step % log_step == 0:
                        loss = accelerator.reduce(loss, "mean ")
                        accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                        accelerator.log({"loss": loss.item()}, step=global_step)

                    if global_step % 50 == 0:
                        # 保存检查点
                        accelerator.print(f"Saving checkpoint -> step_{global_step}")
                        accelerator.save_state(accelerator.project_dir + f"/step_{global_step}")
                        # accelerator.save_model(model, accelerator.project_dir + f"/step_{global_step}")  # 不会保存config.json
                        accelerator.unwrap_model(model).save_pretrained(
                            save_directory=accelerator.project_dir + f"/step_{global_step}/model",
                            is_main_process=accelerator.is_main_process,
                            state_dict=accelerator.get_state_dict(model),
                            save_function=accelerator.save,
                        )

                # if global_step % log_step == 0:
                #     loss = accelerator.reduce(loss, "mean")
                #     accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                # global_step += 1
        acc = evaluate(model, validloader, accelerator) 
        accelerator.print(f"ep: {ep}, acc: {acc}, time: {time.time() - start_time:.2f}")
        accelerator.log({"acc": acc}, step=global_step)
    

def main():
    # accelerator = Accelerator()  # 第二行改动
    # accelerator = Accelerator(mixed_precision="bf16")  # 混合精度

    # 梯度累积在DeepSpeed 0.16.0中不能用，改为1
    accelerator = Accelerator(gradient_accumulation_steps=1, log_with="tensorboard", project_dir="ckpts")  # 梯度累积

    accelerator.init_trackers("runs")

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    # 报错tensor不连续后修改
    for param in model.parameters():
        param.data = param.data.contiguous()

    model, optimizer, trainloader, validloader = accelerator.prepare(model, optimizer, trainloader, validloader)  # 第三行改动

    # train(model, optimizer, trainloader, validloader, accelerator, resume="/home/work/tanshuai/ts/hello_huggingface/05分布式训练篇/ckpts/step_100")
    train(model, optimizer, trainloader, validloader, accelerator, resume=None)
    

    accelerator.end_training()

    if dist.is_initialized():
        dist.destroy_process_group()  # 销毁分布式进程组


if __name__ == "__main__":
    main()
