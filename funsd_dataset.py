import torch
from torch.utils.data import Dataset

class FunsdDataset(Dataset):
    def __init__(self, txt_file, tokenizer, label_list, max_len=512):
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_len = max_len
        self.samples = self._parse_file(txt_file)

    def _parse_file(self, txt_file):
        samples = []
        with open(txt_file, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")
            for block in blocks:
                words, boxes, labels = [], [], []
                for line in block.split("\n"):
                    splits = line.split("\t")
                    if len(splits) != 3:
                        continue
                    word, box, label = splits
                    words.append(word)
                    boxes.append(list(map(int, box.split())))
                    labels.append(label)
                samples.append((words, boxes, labels))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, boxes, labels = self.samples[idx]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        bbox, label_ids = [], []
        prev_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                bbox.append([0, 0, 0, 0])
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                bbox.append(boxes[word_idx])
                label_ids.append(self.label2id.get(labels[word_idx], 0))  # default "O"
            else:
                bbox.append(boxes[word_idx])
                label_ids.append(-100)
            prev_word_idx = word_idx

        encoding["bbox"] = torch.tensor(bbox)
        encoding["labels"] = torch.tensor(label_ids)
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding
