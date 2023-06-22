from torch.utils.data import Dataset, DataLoader
from utils import norm_text, preprocess_text
import torch

class Dual_Infer_Dataset(Dataset):
    def __init__(self, texts, tokenizer, segmenter, max_length=256):
        super().__init__()
        self.texts = texts
        self.tokenizer = tokenizer
        self.segmenter = segmenter
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def pad_1d(self, input_ids, max_length=None):
        if max_length is None:
            max_length = max([len(sample) for sample in input_ids])        
            
        attention_masks = []
        for i in range(len(input_ids)):
            if input_ids[i].size(0) < max_length:
                input_ids[i] = torch.cat(
                    (
                        input_ids[i], 
                        torch.Tensor([self.tokenizer.pad_token_id, ]*(max_length-len(input_ids[i])))
                        )
                    )
            attention_masks.append(input_ids[i] != self.tokenizer.pad_token_id)
            
        return {
            "input_ids": torch.vstack(input_ids),
            "attention_mask": torch.vstack(attention_masks)
        }
    
    def __getitem__(self, index):
        raw_text = self.texts[index]
        
        preprocessed_text = preprocess_text(raw_text)
        segmented_text = " ".join(self.segmenter.word_segment(preprocessed_text))
        normed_text = norm_text(segmented_text)
        
        text_ids = self.tokenizer(
            normed_text, max_length=self.max_length, truncation=True)["input_ids"]

        return {
            "raw_text":raw_text,
            "text_ids":text_ids
        }
    
    def dual_collate_fn(self, batch):
        _input_ids = [
            torch.Tensor(sample["text_ids"])
            for sample in batch 
        ]
        _raw_texts= [
            sample["raw_text"]
            for sample in batch 
        ]
        
        
        padded_input = self.pad_1d(_input_ids)

        input_ids = padded_input["input_ids"].long()
        attention_mask = padded_input["attention_mask"].bool()
                
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "texts":_raw_texts
        }
    