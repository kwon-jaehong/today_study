import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from d_set import CustomDataset
from torch.utils.data import DataLoader, random_split
import pandas as pd
from model import PARSeq
from torch.optim.lr_scheduler import OneCycleLR
import torch
import torch.nn.functional as F



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

train_transform = A.Compose([
                        A.Resize(32,128),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])
# df,transforms
df = pd.read_csv('../train.csv')
data_path = "../"
train_dataset = CustomDataset(df,data_path,train_transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 4,pin_memory=True,shuffle=True)


model = PARSeq()
model.load_state_dict(torch.load('./model.pth'))
model.to(device=device)
model.set_device(device=device)
model.create_optimizer()

epochs = 50
scheduler = OneCycleLR(model.optim, 0.00035, epochs*(len(train_dataset)*batch_size), pct_start=0.075,cycle_momentum=False)


for epoch in range(0,epochs):
    
    model.train()
    for images, labels  in train_loader:
        
        # model.forward_logits_loss(images,labels)
        
        
        
        
        
        model.optim.zero_grad(set_to_none=True)
        
        images = images.to(device=device)    
        tgt = model.tokenizer.encode(labels, device)
        memory = model.encode(images)
        
        # Prepare the target sequences (input and output)
        # 대상 시퀀스 준비(입력 및 출력)
        tgt_perms = model.gen_tgt_perms(tgt)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        # The [EOS] token is not depended upon by any other token in any permutation ordering
        # [EOS] 토큰은 순열 순서에서 다른 토큰에 의존하지 않습니다.
        tgt_padding_mask = (tgt_in == model.pad_id) | (tgt_in == model.eos_id)
        
        


    
        loss = 0
        loss_numel = 0
        n = (tgt_out != model.pad_id).sum().item()
        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = model.generate_attn_masks(perm)
            out = model.decode(tgt_in, memory, tgt_mask, tgt_padding_mask, tgt_query_mask=query_mask)
            logits = model.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=model.pad_id)
            loss_numel += n
            # After the second iteration (i.e. done with canonical and reverse orderings),
            # remove the [EOS] tokens for the succeeding perms
            # 두 번째 반복 후(즉, 정식 및 역순으로 완료),
            # 후속 perms에 대한 [EOS] 토큰을 제거합니다.
            if i == 1:
                tgt_out = torch.where(tgt_out == model.eos_id, model.pad_id, tgt_out)
                n = (tgt_out != model.pad_id).sum().item()
            loss /= loss_numel
        loss.backward()
        model.optim.step()                
        scheduler.step()
        
        
        logits = model.forward(images)
        probs = logits.softmax(-1)
        preds, probs = model.tokenizer.decode(probs)
        # print(preds)
        
        print(loss.item(),labels,preds)


print(0)