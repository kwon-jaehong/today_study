import random
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import f1_score
from konlpy.tag import Mecab
import mlflow
import re
from imblearn.over_sampling import RandomOverSampler 
import random
import pickle

# 토치 텍스트 관련 참조자료
# https://towardsdatascience.com/custom-datasets-in-pytorch-part-2-text-machine-translation-71c41a3e994e

os.chdir('../')
CFG = {
    'IMG_SIZE':128,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':64,
    'SEED':41,
    'MAX_VOCAB_SIZE':100000,
    'TRAIN_RATE':0.9,
    'NUM_WORKERS':4,
    'freq_threshold':0
}



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(CFG['SEED']) # Seed 고정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


wordnet = {}
with open("./wordnet.pickle", "rb") as f:
	wordnet = pickle.load(f)

# 한글만 남기고 나머지는 삭제
def get_only_hangul(line):
	parseText= re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('',line)

	return parseText

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################
def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1
		if num_replaced >= n:
			break

	if len(new_words) != 0:
		sentence = ' '.join(new_words)
		new_words = sentence.split(" ")

	else:
		new_words = ""

	return new_words


def get_synonyms(word):
	synomyms = []

	try:
		for syn in wordnet[word]:
			for s in syn:
				synomyms.append(s)
	except:
		pass

	return synomyms

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################
def random_deletion(words, p):
	if len(words) == 1:
		return words

	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)

	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0

	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words

	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################
def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	
	return new_words


def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		if len(new_words) >= 1:
			random_word = new_words[random.randint(0, len(new_words)-1)]
			synonyms = get_synonyms(random_word)
			counter += 1
		else:
			random_word = ""

		if counter >= 10:
			return
		
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

def EDA(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	sentence = get_only_hangul(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word is not ""]
	num_words = len(words)

	augmented_sentences = []
	num_new_per_technique = int(num_aug/4) + 1

	n_sr = max(1, int(alpha_sr*num_words))
	n_ri = max(1, int(alpha_ri*num_words))
	n_rs = max(1, int(alpha_rs*num_words))

	# sr
	for _ in range(num_new_per_technique):
		a_words = synonym_replacement(words, n_sr)
		augmented_sentences.append(' '.join(a_words))

	# ri
	for _ in range(num_new_per_technique):
		a_words = random_insertion(words, n_ri)
		augmented_sentences.append(' '.join(a_words))

	# rs
	for _ in range(num_new_per_technique):
		a_words = random_swap(words, n_rs)
		augmented_sentences.append(" ".join(a_words))

	# rd
	for _ in range(num_new_per_technique):
		a_words = random_deletion(words, p_rd)
		augmented_sentences.append(" ".join(a_words))

	augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]
	random.shuffle(augmented_sentences)

	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	augmented_sentences.append(sentence)

	return augmented_sentences

def while_EDA(sentence):
    count = 0
    while count < 100:
        aug_list = EDA(sentence=sentence,num_aug=10)
        for aug in aug_list:
            if aug != sentence:
                return aug
        count +=1
    return aug
# while_EDA("제가 우울감을 느낀지는 오래됐는데 점점 개선되고 있다고 느껴요")
# while_EDA("집가고싶다")

def make_aug_data_df(df,index_list,target_quantity):
    ## df 칼럼 복사 data frame
    result_df = df.iloc[0:0]
    
    ## 순차적으로 데이터를 증강함
    for i in range(0,target_quantity - len(index_list)):
        add_sentance = while_EDA(df.iloc[index_list[i%len(index_list)]]['overview'])
        temp = df.iloc[index_list[i%len(index_list)]]
        temp['overview'] = add_sentance
        result_df.loc[len(result_df)+1] = temp
    # df_temp.append(dict(temp),ignore_index=True)

    return result_df


def custom_oversampling(df,label_name,rank_number):
    df = df.reset_index(drop=True)
    label_len = list(df.groupby(label_name).size())

    ## 모든 데이터가 증강할 수량
    target_quantity = sorted(label_len,reverse=True)[rank_number]

    label_list = list(set(list(df[label_name])))

    index_dict = {label_name:[] for label_name in label_list}

    for i,row in enumerate(df[label_name]):
        index_dict[row].append(i)



    aug_df = df.iloc[0:0]
    for key,item in tqdm(index_dict.items(),desc="데이터셋 샘플링중"):
        if len(item) < target_quantity:
            aug_ = make_aug_data_df(df,item,target_quantity)
            aug_df = pd.concat([aug_df,aug_])

    df = pd.concat([df,aug_df])

    return df



def split_df(df,train_rate,min_num,label_column):

    label_list = list(set(list(df[label_column])))
    label_list.sort()


    ## 각 항목별 df의 index가져옴
    train_index_list = []
    val_index_list = []
    for label in label_list:
        index_list = list(df[df[label_column]==label].index)
        ## 필요하다면 여기서 인덱스 리스트를 셔플해도 됨
        
        index_len = len(list(df[df[label_column]==label].index))

        if index_len*train_rate > min_num:
            train_index_list = train_index_list + index_list[:int(index_len*train_rate)]
            val_index_list = val_index_list + index_list[int(index_len*train_rate):]

    train_df = df.iloc[train_index_list]

    val_df = df.iloc[val_index_list]

    return train_df,val_df
    



class Vocabulary:
    def __init__(self, freq_threshold, max_size,mecab):
        self.itos = {0: '<unk>', 1:'<pad>'}
        self.stoi = {k:j for j,k in self.itos.items()} 
        self.mecab = mecab
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        self.bi_gram =True
    
    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 2
        self.max_len = 0

        
        for sentence in sentence_list:
            sentence = self.mecab.morphs(sentence)
            
            ## 바이그램으로 구성할지 여부
            if self.bi_gram:
                sentence = self.generate_bigrams(sentence)
            
            ## 문장 최대 길이
            if len(sentence) > self.max_len:
                self.max_len = len(sentence)
            
            for word in sentence:
                if word not in frequencies.keys():
                    frequencies[word]=1
                else:
                    frequencies[word]+=1
        frequencies = {k:v for k,v in frequencies.items() if v>self.freq_threshold} 
        frequencies = dict(sorted(frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx]) # idx =4 for pad, start, end , unk
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
            
    def numericalize(self, text):
        tokenized_text = self.mecab.morphs(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else:
                numericalized_text.append(self.stoi['<unk>'])
                
        return numericalized_text
    
    ## bi그램 처리
    def generate_bigrams(self,x):
        n_grams = set(zip(*[x[i:] for i in range(2)]))
        for n_gram in n_grams:
            x.append(' '.join(n_gram))
        return x
    
    
    
class CustomDataset(Dataset):
    def __init__(self,df,vocabulary,lable2num,transforms,infer=False):
        self.all_df = df    
        self.vocabulary = vocabulary

        self.text_list = list(self.all_df['overview'])
        for i,text in enumerate(self.text_list):
            self.text_list[i] = self.cleanhtml(text)
            
        self.img_path_list = list(self.all_df['img_path'])
        self.infer = infer
        if infer ==False:
            self.label_list = list(self.all_df['cat3'])
            self.lable2num = lable2num      

        self.transforms = transforms
        
        
    def __getitem__(self, index):
        text = self.text_list[index]
        text_vector = self.vocabulary.numericalize(text)
        text_len = len(text_vector)
        
        ## Image
        img_path = self.img_path_list[index]
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        # Label
        if self.infer:
            return image, torch.Tensor(text_vector).view(-1),torch.tensor([text_len],dtype=torch.long)
        else:
            label = self.lable2num[self.label_list[index]]
            return image, torch.Tensor(text_vector).view(-1).to(torch.long), torch.tensor([label],dtype=torch.long),torch.tensor([text_len],dtype=torch.long)

    def __len__(self):
        return len(self.all_df)
        
    def cleanhtml(self,raw_html):
        cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        cleantext = re.sub(cleanr, '', raw_html)
        return cleantext



def score_function(real, pred):
    return f1_score(real, pred, average="weighted")


class MyCollate:
    def __init__(self, pad_idx,infer=False):
        self.pad_idx = pad_idx
        self.infer = infer

    def __call__(self, batch):
        image = [item[0] for item in batch]  
        text = [item[1] for item in batch]  
        text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value = self.pad_idx) 
        if self.infer == False:
            label = [item[2] for item in batch]  
            text_len = [item[3] for item in batch]  
            return {"image":torch.stack(image),"text":text,"label":torch.stack(label).squeeze(),"text_len":torch.stack(text_len)}
        else:
            text_len = [item[2] for item in batch]  
            return {"image":torch.stack(image),"text":text,"text_len":torch.stack(text_len)}


class CustomModel(nn.Module):
    def __init__(self,vocab_size, num_classes):
        super(CustomModel, self).__init__()
        
        self.embedding_dim = 1024
        self.vocab_size = vocab_size
        # Image
        self.cnn_extract = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 이미지만 3136
        
        # Text
        self.bon_embed = nn.Embedding(vocab_size,self.embedding_dim,padding_idx=1)
        self.fc = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.nlp_bn = nn.BatchNorm1d(self.embedding_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(4160, num_classes)
        )
            

    def forward(self, img, text,text_len):
        img_feature = self.cnn_extract(img)
        img_feature = torch.flatten(img_feature, start_dim=1)
        
        embed = self.bon_embed(text)
        embed = torch.sum(embed, 1).squeeze(1)
        batch_size = embed.size(0)
        text_len = text_len.expand(batch_size, self.embedding_dim)
        embed /= text_len
        embed = self.nlp_bn(embed)
        text_feature = self.fc(embed)
        
        feature = torch.cat([img_feature, text_feature], axis=1)
        
        
        output = self.classifier(feature)
        return output


mlflow.set_experiment('fast_text')
with mlflow.start_run() as run:
    mlflow.log_artifact(__file__)
    mlflow.log_params(CFG)
    train_transform = A.Compose([
                                A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])
    test_transform = A.Compose([
                                A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])
    
    all_df = pd.read_csv('./train.csv')
    
    label_list = list(all_df['cat3'])
    
    set_label_list = list(set(label_list))
    set_label_list.sort()
    
    num2label = {i:label for i,label in enumerate(set_label_list)}
    lable2num = {label:i for i,label in enumerate(set_label_list)}
    label_info = lable2num
    
    text_list = list(all_df['overview'])
    TEXT = Vocabulary(CFG["freq_threshold"],CFG['MAX_VOCAB_SIZE'],Mecab())
    TEXT.build_vocabulary(text_list)      
    vocab_size = len(TEXT.itos)  
    print("단어사전 구성 끝")
   

    
    print("데이터셋 구성 완료")
    train_df,val_df = split_df(all_df,CFG['TRAIN_RATE'],1,"cat3")

    ## 오버 
    # temp_df = train_df.copy()
    train_df = custom_oversampling(train_df,'cat3',5)
    # y = train_df['cat3']
    # ros = RandomOverSampler(random_state=0)
    # sampleing_df = ros.fit_resample(temp_df, y)
    

    train_dataset = CustomDataset(train_df,TEXT,lable2num,train_transform)
    validation_dataset = CustomDataset(val_df,TEXT,lable2num,train_transform)
    
    

    pad_idx = TEXT.stoi['<pad>']
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],shuffle=True,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx))
    validation_loader = DataLoader(validation_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],shuffle=True,pin_memory=True, collate_fn = MyCollate(pad_idx=pad_idx))
        




    model = CustomModel(vocab_size,len(label_info))
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    best_score = 0
    best_model = None
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])




    for epoch in range(0,CFG["EPOCHS"]):
        model.train()
        train_loss = []   
        train_data_len = train_dataset.__len__()
        total_train_correct = 0
        
        for i,data_batch in enumerate(train_loader):
            img = data_batch['image']
            text = data_batch['text']
            label = data_batch['label']
            text_len = data_batch['text_len']
            
            img = img.float().to(device)
            text = text.to(device)
            label = label.to(device)
            text_len = text_len.to(device)
            optimizer.zero_grad()
            
            model_pred = model(img, text,text_len)        
            loss = criterion(model_pred, label)
            loss.backward()
            optimizer.step()                
            _, predicted = torch.max(model_pred, 1) 
            correct = (predicted == label).sum().item()        
            total_train_correct += correct        
            train_loss.append(loss.item())
            
            print(f'epoch {epoch}/{CFG["EPOCHS"]+1} {(i*CFG["BATCH_SIZE"])+len(label)}/{train_data_len} train loss {loss.item():.4f} acc : {100*correct/len(label):.2f}% - ({correct}/{len(label)})')
            
        tr_loss = np.mean(train_loss)
        print(f"\n epoch {epoch} train end!!! \t train batch loss : {tr_loss:.4f}\t total acc : {100*total_train_correct/train_data_len:.2f}% - ({total_train_correct}/{train_data_len}) \n")
        mlflow.log_metric("train_loss",np.mean(train_loss))
        mlflow.log_metric("train_acc",100*total_train_correct/train_data_len)

        
        
        val_data_len = validation_dataset.__len__()
        model.eval()   
        model_preds = []
        true_labels = []    
        val_loss = []    
        total_val_correct = 0
        with torch.no_grad():
            for i,data_batch in enumerate(validation_loader):
                img = data_batch['image']
                text = data_batch['text']
                label = data_batch['label']
                text_len = data_batch['text_len']
            
                img = img.float().to(device)
                text = text.to(device)
                label = label.to(device)
                text_len = text_len.to(device)
                
                model_pred = model(img, text,text_len)
                
                loss = criterion(model_pred, label)
                
                _, predicted = torch.max(model_pred, 1) 
                correct = (predicted == label).sum().item() 
                total_val_correct+=correct
                val_loss.append(loss.item())
                
                model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
                true_labels += label.detach().cpu().numpy().tolist()
            
        test_weighted_f1 = score_function(true_labels, model_preds)
        
        print(f"epoch {epoch} val end!!! val loss : {np.mean(val_loss):.3f} \t f1 score : {test_weighted_f1:.3f} acc : {100*total_val_correct/val_data_len:.2f}% - ({total_val_correct}/{val_data_len}) \n\n")    

        mlflow.log_metric("weighted_f1",test_weighted_f1)
        mlflow.log_metric("val_loss",np.mean(val_loss))
        mlflow.log_metric("val_acc",100*total_val_correct/val_data_len)

























