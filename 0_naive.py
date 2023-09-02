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

        similarity = Q @ K.T
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

source_tokens = list(source_tokens)
target_tokens = list(target_tokens)


dataset_numeric = []
for data in dataset:
    en_seq = data["translation"]["en"]
    de_seq = data["translation"]["de"]

    numeric_data = {}
    numeric_data["en"] = [source_tokens.index(token) for token in en_seq.split()]
    numeric_data["de"] = [target_tokens.index(token) for token in de_seq.split()]

    numeric_data["de"].insert(0, target_tokens.index("<EOS>"))
    numeric_data["de"].append(target_tokens.index("<EOS>"))

    dataset_numeric.append(numeric_data)


transformer = Transformer(len(source_tokens), len(target_tokens), embed_dim=4)
optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-3)

transformer.train()
for epoch in range(200):
    losses = []
    for data in dataset_numeric:
        src_seq = torch.tensor(data["en"])
        tgt_seq = torch.tensor(data["de"])

        first_n_token = torch.randint(low=1, high=len(tgt_seq), size=(1,)).item()

        output = transformer(src_seq, tgt_seq[:first_n_token])
        loss = F.cross_entropy(output, tgt_seq[1 : first_n_token + 1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f"Epoch: {epoch+1}, Loss: {sum(losses)}")


transformer.eval()
with torch.no_grad():
    for data in dataset_numeric:
        src_seq = torch.tensor(data["en"])

        translation = torch.tensor([target_tokens.index("<EOS>")])
        translation_text = []

        for _ in range(40):  # we say it can be max 40 tokens per translation
            next_word_prob = transformer(src_seq, translation)[-1]
            pred = torch.argmax(next_word_prob).item()
            next_word = target_tokens[pred]

            translation_text.append(next_word)
            translation = torch.cat(
                (
                    translation,
                    torch.tensor([target_tokens.index(next_word)]),
                )
            )

            if next_word == "<EOS>":
                break

        en_text = " ".join([source_tokens[idx] for idx in data["en"]])
        de_text = " ".join([target_tokens[idx] for idx in data["de"][1:]])
        de_pred = " ".join(translation_text)

        print(f"orig: {en_text}")
        print(f"real: {de_text}")
        print(f"pred: {de_pred}")
        print("---------")
