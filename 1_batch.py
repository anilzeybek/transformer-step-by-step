import datasets
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)


class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, y, z):
        Q = self.W_q(x)
        K = self.W_k(y)
        V = self.W_v(z)

        similarity = Q @ K.transpose(1, 2)
        attention = torch.softmax(similarity, dim=-1) @ V

        return attention


class Transformer(nn.Module):
    def __init__(self, source_vocab_len, target_vocab_len, embed_dim):
        super().__init__()

        self.encoder_embedding = nn.Embedding(source_vocab_len, embed_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_len, embed_dim)

        self.encoder_attention = Attention(embed_dim)
        self.decoder_attention = Attention(embed_dim)
        self.cross_attention = Attention(embed_dim)

        self.fc = nn.Linear(embed_dim, target_vocab_len)

    def forward(self, source_seq, target_seq):
        source_embed = self.encoder_embedding(source_seq)
        target_embed = self.decoder_embedding(target_seq)

        encoder_output = self.encoder_attention(source_embed, source_embed, source_embed)
        encoder_output += source_embed

        decoder_output = self.decoder_attention(target_embed, target_embed, target_embed)
        decoder_output += target_embed

        cross_output = self.cross_attention(decoder_output, encoder_output, encoder_output)
        cross_output += decoder_output

        return self.fc(cross_output)


dataset = datasets.load_dataset("opus100", "de-en")["train"].select(range(50))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

source_tokens = set()
target_tokens = set()

for data in dataset:
    source = data["translation"]["en"]
    target = data["translation"]["de"]

    s_token_list = source.split()
    t_token_list = target.split()

    source_tokens.update(s_token_list)
    target_tokens.update(t_token_list)

source_tokens.add("<EOS>")
target_tokens.add("<EOS>")

source_tokens.add("<PAD>")
target_tokens.add("<PAD>")

source_tokens = list(source_tokens)
target_tokens = list(target_tokens)


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


device = "cuda" if torch.cuda.is_available() else "cpu"

transformer = Transformer(len(source_tokens), len(target_tokens), embed_dim=4).to(device)
optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-3)

transformer.train()
for epoch in range(1000):
    losses = []
    for data in dataloader:
        src_seq, tgt_seq = get_numeric_data(data, source_tokens, target_tokens)
        src_seq = src_seq.to(device)
        tgt_seq = tgt_seq.to(device)

        first_n_token = torch.randint(low=1, high=tgt_seq.shape[1], size=(1,)).item()

        output = transformer(src_seq, tgt_seq[:, :first_n_token])
        loss = F.cross_entropy(
            output.view(-1, len(target_tokens)),
            tgt_seq[:, 1 : first_n_token + 1].contiguous().view(-1),
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
            next_word_probs = transformer(src_seq, translations)[:, -1, :]
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
