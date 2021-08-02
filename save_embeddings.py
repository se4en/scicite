import jsonlines
from transformers import *
import torch
import numpy as np

FILE_TEST_TRUE = "scicite/data/acl-arc/test.jsonl"
FILE_DEV_TRUE = "scicite/data/acl-arc/dev.jsonl"
FILE_TRAIN_TRUE = "scicite/data/acl-arc/train.jsonl"

FILE_TEST_NEW = "scicite/data/acl-arc-emb/test.jsonl"
FILE_DEV_NEW = "scicite/data/acl-arc-emb/dev.jsonl"
FILE_TRAIN_NEW = "scicite/data/acl-arc-emb/train.jsonl"

FILE_TRUE_EMBS = "scicite/data/acl-arc-emb/embeddings.npy"

FILE_TRAIN_TRUE_SCAF_1 = "scicite/data/acl-arc/scaffolds/cite-worthiness-scaffold-train.jsonl"
FILE_TRAIN_TRUE_SCAF_1_NEW = "scicite/data/acl-arc-emb/scaffolds/cite-worthiness-scaffold-embs/"

FILE_TRAIN_TRUE_SCAF_2 = "scicite/data/acl-arc/scaffolds/sections-scaffold-train.jsonl"
FILE_TRAIN_TRUE_SCAF_2_NEW = "scicite/data/acl-arc-emb/scaffolds/sections-scaffold-embs/"

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", do_lower_case=True)
model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")


def save_embs():
    emb_cnt = 0
    embeddings = []

    with jsonlines.open(FILE_TEST_TRUE, 'r') as reader:
        with jsonlines.open(FILE_TEST_NEW, 'w') as writer:
            for item in enumerate(reader):
                tok_out = tokenizer.encode(item[1]['text'], padding='max_length', max_length=400)
                tok_out = torch.tensor(tok_out).unsqueeze(0)
                # print(tok_out.size())
                emb_out = model(tok_out)[0]

                item[1]['emb_id'] = emb_cnt
                embeddings.append(emb_out.detach().numpy())
                writer.write(item[1])
                emb_cnt += 1
    print("done_test")

    with jsonlines.open(FILE_DEV_TRUE, 'r') as reader:
        with jsonlines.open(FILE_DEV_NEW, 'w') as writer:
            for item in enumerate(reader):
                tok_out = tokenizer.encode(item[1]['text'], padding='max_length', max_length=400)
                emb_out = model(torch.tensor(tok_out).unsqueeze(0))[0]

                item[1]['emb_id'] = emb_cnt
                embeddings.append(emb_out.detach().numpy())
                writer.write(item[1])
                emb_cnt += 1
    print("done_dev")

    with jsonlines.open(FILE_TRAIN_TRUE, 'r') as reader:
        with jsonlines.open(FILE_TRAIN_NEW, 'w') as writer:
            for item in enumerate(reader):
                tok_out = tokenizer.encode(item[1]['text'], padding='max_length', max_length=400)
                emb_out = model(torch.tensor(tok_out).unsqueeze(0))[0]

                item[1]['emb_id'] = emb_cnt
                embeddings.append(emb_out.detach().numpy())
                writer.write(item[1])
                emb_cnt += 1
    print("done_train")

    np.save(FILE_TRUE_EMBS, np.array(embeddings))


if __name__ == "__main__":
    save_embs()
