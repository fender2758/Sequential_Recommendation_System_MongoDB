import wandb
from transformers import TrainerCallback
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import ndcg_score
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import BertConfig, BertForMaskedLM
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast

from transformers.models.bert.modeling_bert import BertLMPredictionHead
import torch.nn as nn

class ItemHistoryDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        user_history = self.dataframe.iloc[index]['item_history']
        inputs = self.tokenizer(
            user_history,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs


def compute_hit_ratio(scores, true_labels, k):
    top_k = np.argsort(scores)[-k:]
    return np.isin(np.where(true_labels == 1)[0], top_k).sum()

def compute_ndcg(scores, true_labels, k):
    return ndcg_score([true_labels], [scores], k=k)

#def compute_ndcg(scores, true_labels, k):
#    rank = scores.argsort()[0]
#    return 1/np.log2(rank+2) if rank<k else 0

#def compute_hit_ratio(predictions, labels, k):
#    top_k = np.argsort(predictions)[-k:]
#    return int(labels[-1] in top_k)

def compute_mrr(predictions, labels):
    match = np.where(np.argsort(-predictions) == labels[-1])
    rank = match[0][0] + 1
    return 1 / rank


class EvaluationMetricsCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        if (state.epoch + 1) % 10 != 0:
            return
        trainer = kwargs['model']
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=8)
        ndcg_1_scores, ndcg_5_scores, ndcg_10_scores = [], [], []
        hit_ratio_1_scores, hit_ratio_5_scores, hit_ratio_10_scores = [], [], []
        mrr_scores = []
        
        for batch in eval_dataloader:
            inputs = {k: v.to(trainer.device) for k, v in batch.items()}
            mask_token_index = []
            labels = []
            for i, input in enumerate(inputs['input_ids']):
                try:
                    idx = torch.where(input==0)[0][0]-1
                    mask_token_index.append(idx.cpu().item())
                except:
                    idx=-1
                    mask_token_index.append(idx)
                labels.append(input[idx].cpu().item())
                input[idx]=self.tokenizer.mask_token_id
                inputs['input_ids'][i]=input
            with torch.no_grad():
                outputs = trainer(**inputs)
                #print(outputs.logits)
                random_sampled = []
                all_items = set(range(len(self.tokenizer)))
                neg_items = 100
                mask_token_logits = outputs.logits[np.arange(len(inputs['input_ids'])), mask_token_index, :]
                for input_ids in inputs['input_ids']:
                    # Convert input_ids to a set for faster exclusion
                    input_ids_set = set(input_ids.tolist())
                    
                    # Get the set difference
                    possible_negatives = list(all_items - input_ids_set)
                    #possible_negatives_tensor = torch.tensor(possible_negatives, dtype=torch.float32)

                    # Sample neg_items-1 items from the possible negatives
                    #negative_samples = possible_negatives_tensor.multinomial(neg_items - 1, replacement=False)
                    negative_samples = np.random.choice(possible_negatives, neg_items - 1)
                    
                    random_sampled.append(torch.from_numpy(negative_samples))

                #random_sampled = torch.randint(len(self.tokenizer), (len(inputs['input_ids']), neg_items-1))
                random_sampled = torch.stack(random_sampled)
                combined = torch.cat((torch.tensor(labels).unsqueeze(-1), random_sampled), dim=-1)

                #labels = inputs['input_ids'].cpu().numpy()[np.arange(len(inputs['input_ids'])), mask_token_index]
                #predictions = torch.argmax(mask_token_logits, dim=-1).cpu().numpy()
                predictions = combined
                length = len(inputs['input_ids'])
                scores = mask_token_logits[np.arange(length).reshape(length, 1) * np.ones(neg_items), combined].cpu().numpy()
                #print(predictions, labels)
                real_labels = [np.where(preds==l, 1, 0) for preds, l in zip(predictions, labels)]
                for pred, label in zip(scores, real_labels):
                    #print(pred, label)
                    ndcg_1_scores.append(compute_ndcg(pred, label, k=1))
                    ndcg_5_scores.append(compute_ndcg(pred, label, k=5))
                    ndcg_10_scores.append(compute_ndcg(pred, label, k=10))
                    hit_ratio_1_scores.append(compute_hit_ratio(pred, label, k=1))
                    hit_ratio_5_scores.append(compute_hit_ratio(pred, label, k=5))
                    hit_ratio_10_scores.append(compute_hit_ratio(pred, label, k=10))
                    mrr_scores.append(compute_mrr(pred, label))
            
        #print(ndcg_10_scores[:100])
        avg_ndcg_1 = np.mean(ndcg_1_scores)
        avg_ndcg_5 = np.mean(ndcg_5_scores)
        avg_ndcg_10 = np.mean(ndcg_10_scores)
        avg_hit_ratio_1 = np.mean(hit_ratio_1_scores)
        avg_hit_ratio_5 = np.mean(hit_ratio_5_scores)
        avg_hit_ratio_10 = np.mean(hit_ratio_10_scores)
        avg_mrr = np.mean(mrr_scores)

        wandb.log({
            "NDCG@1": avg_ndcg_1,
            "NDCG@5": avg_ndcg_5,
            "NDCG@10": avg_ndcg_10,
            "Hit Ratio@1": avg_hit_ratio_1,
            "Hit Ratio@5": avg_hit_ratio_5,
            "Hit Ratio@10": avg_hit_ratio_10,
            "MRR": avg_mrr
        })

def main(args):
    wandb.init(project=args.project, name=args.run_name+"NLLloss")

    df = pd.read_csv('data/Beauty.csv', names=['user','item','time','rating'], skiprows=1)

    unique_tokens = set(token for token in df['item'])

    # Create a mapping from original tokens to unique IDs
    vocab = {'item'+str(token): idx+2 for idx, token in enumerate(sorted(unique_tokens), 1)}  # Starting index from 1
    vocab['[PAD]']=0
    vocab['[UNK]']=1
    vocab['[MASK]']=2
    # Save the vocab to a file (optional)
    import json
    vocab_file = 'vocab.json'
    with open(vocab_file, 'w') as file:
        json.dump(vocab, file)


    # Initialize tokenizer

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    #tokenizer.decoder = decoders.WordPiece(prefix="##")

    # Save the tokenizer
    tokenizer_file = "custom_tokenizer.json"
    tokenizer.save(tokenizer_file)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, unk_token="[UNK]", max_length=128)

    special_tokens = {'pad_token': '[PAD]', 'mask_token': '[MASK]'}
    tokenizer.add_special_tokens(special_tokens)

    df = df.sort_values(by=['user', 'time'])

    def has_at_least_5_items(row):
        items = row.split()
        return len(items) >= 5
    


    # Create user history as a sequence of item IDs
    user_histories = df.groupby('user')['item'].apply(lambda x: 'item' + ' item'.join(map(str, x[:-1]))).reset_index()
    user_histories.columns = ['user', 'item_history']

    # Filter rows based on the number of items in 'item_history'
    user_histories = user_histories[user_histories['item_history'].apply(has_at_least_5_items)].reset_index(drop=True)
    
    # Create dataset
    train_dataset = ItemHistoryDataset(user_histories, tokenizer, max_len=args.context)

    # Create user history as a sequence of item IDs
    user_histories = df.groupby('user')['item'].apply(lambda x: 'item' + ' item'.join(map(str, x))).reset_index()
    user_histories.columns = ['user', 'item_history']


    # Filter rows based on the number of items in 'item_history'
    user_histories = user_histories[user_histories['item_history'].apply(has_at_least_5_items)].reset_index(drop=True)

    # Create dataset
    eval_dataset = ItemHistoryDataset(user_histories, tokenizer, max_len=args.context)

    # Split dataset into train and eval
    #train_size = int(0.8 * len(dataset))
    #train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])


    # Define the model configuration
    config = BertConfig(
            vocab_size = len(tokenizer),
            hidden_size = args.hidden_size,
            max_position_embeddings=args.context,
            attention_probs_dropout_prob=0.2,
            hidden_act='gelu',
            hidden_dropout_prob=0.2,
            initializer_range=0.02,
            num_attention_heads=args.heads,
            num_hidden_layers=args.layers,
            type_vocab_size=2,
    )
    model = BertForMaskedLM(config=config)
    """
    class custom_MLMhead(BertLMPredictionHead):
        def __init__(self, config):
            super().__init__(config)
            self.softmax = nn.Softmax(dim=-1)
        def forward(self, hidden_states):
            hidden_states = self.transform(hidden_states)
            hidden_states = self.decoder(hidden_states)
            hidden_states = self.softmax(hidden_states)
            return hidden_states
            
    model.cls.predictions = custom_MLMhead(config)
    """

    class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
        def __call__(self, examples):
            batch = super().__call__(examples)

            input_ids = batch['input_ids']
            labels = batch['labels']

            for i in range(len(input_ids)):
                # Set the final token as masked
                try:
                    final_token_idx = torch.where(input_ids[i] != self.tokenizer.pad_token_id)[0][-1].item()
                except:
                    final_token_idx = len(input_ids[i])
                if labels[i][final_token_idx] == -100:  # If the final token is not already masked
                    labels[i][final_token_idx] = input_ids[i][final_token_idx]
                    input_ids[i][final_token_idx] = self.tokenizer.mask_token_id

            return batch
        
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.2
    )
    """
    texts = ["item123 item123 item123 item123 item123", "item123 item123 item123 item123 "]
    inputs = tokenizer(texts, padding=True, truncation=False, return_tensors="pt", max_length=64)
    print(inputs)
    # Use the custom data collator
    batch = data_collator([inputs.input_ids[i] for i in range(inputs.input_ids.shape[0])])
    print(batch)
    """
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        #num_train_epochs=10,
        learning_rate= args.lr,
        num_train_epochs=200,
#        max_steps=args.steps,
        warmup_steps=1000,
        per_device_train_batch_size=args.batch,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps",  # Enable evaluation at specific steps
        eval_steps=500,  # Evaluate every 500 steps
    )

    """
    def custom_loss(outputs, labels):
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits)
        loss_fct = torch.nn.NLLLoss(ignore_index=-100)
        log_probs = log_probs.view(-1, log_probs.size(-1)) 
        labels = labels.view(-1)  # [batch_size * seq_length]

        loss = loss_fct(log_probs, labels)
        return loss

    # Custom Trainer class
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # Forward pass
            outputs = model(**inputs)
            # Compute custom loss
            loss = custom_loss(outputs, labels)
            return (loss, outputs) if return_outputs else loss
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EvaluationMetricsCallback(tokenizer, eval_dataset)]
    )

    trainer.train()

    # Save the model
    model.save_pretrained('./models/bert-item-mlm')
    tokenizer.save_pretrained('./models/bert-item-mlm')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with wandb integration")

    parser.add_argument("--project", type=str, default="bert4rec", help="wandb project name")
    parser.add_argument("--run_name", type=str, default="default_run", help="wandb run name")
    parser.add_argument("--lr", type=float, default=5e-05, help="Learning rate for the optimizer")
    parser.add_argument("--steps", type=int, default=60000, help="Number of epochs to train")
    parser.add_argument("--batch", type=int, default=256, help="Number of epochs to train")
    parser.add_argument("--hidden_size", type=int, default=64, help="wandb project name")
    parser.add_argument("--layers", type=int, default=2, help="wandb run name")
    parser.add_argument("--heads", type=int, default=2, help="Learning rate for the optimizer")
    parser.add_argument("--inner_size", type=int, default=256, help="Number of epochs to train")
    parser.add_argument("--context", type=int, default=200, help="Number of epochs to train")

    args = parser.parse_args()
    args.run_name = f'{args.project}-lr{args.lr}-bt{args.batch}-hs{args.hidden_size}-ly{args.layers}-hd{args.heads}-is{args.inner_size}-cl{args.context}'
    main(args)