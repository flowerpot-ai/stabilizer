from torch.utils.data import Dataset


class TextLabelDataset(Dataset):
    def __init__(self, text_excerpts, labels):
        self.text_excerpts = text_excerpts
        self.labels = labels

    def __len__(self):
        return len(self.text_excerpts)

    def __getitem__(self, idx):
        sample = {"text_excerpt": self.text_excerpts[idx], "label": self.labels[idx]}
        return sample
