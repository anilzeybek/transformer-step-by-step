import math

import datasets
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len):
        super().__init__()

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # we use register_buffer instead of self.pe because we want this pe to be in the same device with transformer
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: Tensor, of shape [batch_size, seq_len, embed_dim]
        x = x + self.pe[: x.shape[1]]
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.linear_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y, z, mask=None):
        Q = self.W_q(x)
        K = self.W_k(y)
        V = self.W_v(z)

        similarity = Q @ K.transpose(1, 2)
        if mask is not None:
            similarity = similarity.masked_fill(mask, float("-1e9"))

        attention = torch.softmax(similarity, dim=-1) @ V
        output = self.linear_out(attention)

        return output


class Encoder(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()

        self.attn = Attention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        out = self.attn(x, x, x, mask)
        x = out + x
        x = self.norm1(x)

        out = F.relu(self.ff1(x))
        out = self.ff2(out)

        x = out + x
        x = self.norm2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()

        self.attn = Attention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.cross_attn = Attention(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, y, mask):
        out = self.attn(x, x, x, mask)
        x = out + x
        x = self.norm1(x)

        out = self.cross_attn(x, y, y, mask=None)
        x = out + x
        x = self.norm2(x)

        out = F.relu(self.ff1(x))
        out = self.ff2(out)

        x = out + x
        x = self.norm3(x)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        source_vocab_len,
        target_vocab_len,
        embed_dim,
        num_encoder_layers=6,
        num_decoder_layers=6,
        ff_dim=2048,
        max_seq_len=1024,
    ):
        super().__init__()

        self.encoder_embedding = nn.Embedding(source_vocab_len, embed_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_len, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)

        self.encoders = nn.ModuleList([Encoder(embed_dim, ff_dim=ff_dim) for _ in range(num_encoder_layers)])
        self.decoders = nn.ModuleList([Decoder(embed_dim, ff_dim=ff_dim) for _ in range(num_decoder_layers)])

        self.fc = nn.Linear(embed_dim, target_vocab_len)

    def forward(self, source_seq, target_seq, source_pad_index, target_pad_index):
        source_mask = (source_seq == source_pad_index).unsqueeze(1)
        target_mask = (target_seq == target_pad_index).unsqueeze(1)

        t_seq_len = target_seq.shape[1]
        look_ahead_mask = torch.triu(torch.ones(t_seq_len, t_seq_len, device=device), diagonal=1).bool()
        combined_target_mask = torch.logical_or(target_mask, look_ahead_mask)

        source_embed = self.positional_encoding(self.encoder_embedding(source_seq))
        target_embed = self.positional_encoding(self.decoder_embedding(target_seq))

        encoder_out = source_embed
        for encoder in self.encoders:
            encoder_out = encoder(encoder_out, source_mask)

        decoder_out = target_embed
        for decoder in self.decoders:
            decoder_out = decoder(decoder_out, encoder_out, combined_target_mask)

        return self.fc(decoder_out)


def tokenizer(dataset):
    source_tokens = set()
    target_tokens = set()

    max_seq_len = 0
    for data in dataset:
        s = data["translation"]["en"]
        t = data["translation"]["de"]

        s_token_list = s.split()
        t_token_list = t.split()
        max_seq_len = max(max_seq_len, len(s_token_list), len(t_token_list))

        source_tokens.update(s_token_list)
        target_tokens.update(t_token_list)

    source_tokens.add("<PAD>")
    target_tokens.add("<PAD>")

    source_tokens.add("<EOS>")
    target_tokens.add("<EOS>")

    source_tokens = list(source_tokens)
    target_tokens = list(target_tokens)

    return source_tokens, target_tokens, max_seq_len + 2  # +2 for two <eos> in target sequences


def get_numeric_data(data, source_tokens, target_tokens):
    data = data["translation"]

    max_source_len = 0
    for seq in data["en"]:
        max_source_len = max(max_source_len, len(seq.split()))

    max_target_len = 0
    for seq in data["de"]:
        max_target_len = max(max_target_len, len(seq.split()))

    source_numeric_tokens = []
    target_numeric_tokens = []

    for s_seq, t_seq in zip(data["en"], data["de"]):
        source_numeric_token = []
        tokens = s_seq.split()
        for token in tokens:
            source_numeric_token.append(source_tokens.index(token))

        # padding each sequence
        source_numeric_token = F.pad(
            torch.tensor(source_numeric_token),
            pad=(0, max_source_len - len(source_numeric_token)),
            value=source_tokens.index("<PAD>"),
        )

        source_numeric_tokens.append(source_numeric_token)

        ###

        # we need to have <EOS> at the start and end for target sequences
        target_numeric_token = [target_tokens.index("<EOS>")]

        tokens = t_seq.split()
        for token in tokens:
            target_numeric_token.append(target_tokens.index(token))

        target_numeric_token.append(target_tokens.index("<EOS>"))
        target_numeric_token = F.pad(
            torch.tensor(target_numeric_token),
            pad=(0, max_target_len - len(target_numeric_token)),
            value=target_tokens.index("<PAD>"),
        )

        target_numeric_tokens.append(target_numeric_token)

    return torch.vstack(source_numeric_tokens), torch.vstack(target_numeric_tokens)


dataset = datasets.load_dataset("opus100", "de-en")["train"].select(range(50))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

device = "cuda" if torch.cuda.is_available() else "cpu"

source_tokens, target_tokens, max_seq_len = tokenizer(dataset)

transformer = Transformer(
    len(source_tokens),
    len(target_tokens),
    embed_dim=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    ff_dim=128,
    max_seq_len=max_seq_len,
).to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-3)

transformer.train()
for epoch in range(2000):
    losses = []
    for data in dataloader:
        src_seq, tgt_seq = get_numeric_data(data, source_tokens, target_tokens)
        src_seq = src_seq.to(device)
        tgt_seq = tgt_seq.to(device)

        output = transformer(src_seq, tgt_seq[:, :-1], source_tokens.index("<PAD>"), target_tokens.index("<PAD>"))
        loss = F.cross_entropy(
            output.view(-1, len(target_tokens)),
            tgt_seq[:, 1:].contiguous().view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch+1}, Loss: {sum(losses)}")


transformer.eval()
with torch.no_grad():
    for data in dataloader:
        src_seq, _ = get_numeric_data(data, source_tokens, target_tokens)
        src_seq = src_seq.to(device)

        translations = torch.zeros((src_seq.shape[0], 1), dtype=torch.int64, device=device)
        translations[:] = target_tokens.index("<EOS>")

        translated_texts = []
        for _ in range(40):  # we say it can be max 40 tokens per sequence
            next_word_probs = transformer(
                src_seq, translations, source_tokens.index("<PAD>"), target_tokens.index("<PAD>")
            )[:, -1, :]

            preds = torch.argmax(next_word_probs, dim=-1)
            next_words = [target_tokens[i] for i in preds]
            translated_texts.append(next_words)

            next_tokens = torch.tensor(
                [target_tokens.index(w) for w in next_words],
                dtype=torch.int64,
                device=device,
            ).unsqueeze(1)

            translations = torch.cat((translations, next_tokens), dim=1)

        for i, text_arr in enumerate(list(zip(*translated_texts))):
            if "<EOS>" in text_arr:
                text_arr = text_arr[: text_arr.index("<EOS>") + 1]

            en = data["translation"]["en"][i]
            de = data["translation"]["de"][i]
            de_pred = " ".join(text_arr)

            print(f"orig: {en}")
            print(f"real: {de}")
            print(f"pred: {de_pred}")
            print("---------")
