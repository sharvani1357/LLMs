import torch

# ==========================================
# GREEDY SEARCH
# ==========================================

def generate_text(
        model,
        prompt,
        word2idx,
        idx2word,
        device,
        max_new_tokens=20):

    model.eval()

    words = prompt.lower().split()

    for _ in range(max_new_tokens):

        token_ids = [

            word2idx.get(
                word,
                word2idx["<UNK>"]
            )

            for word in words

        ]

        token_ids = token_ids[-30:]

        padding = [0] * (
            30 - len(token_ids)
        )

        token_ids = (
            padding + token_ids
        )

        input_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long
        ).to(device)

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

        logits = outputs[:, -1, :]

        next_token = torch.argmax(
            logits,
            dim=-1
        ).item()

        next_word = idx2word.get(
            next_token,
            "<UNK>"
        )

        words.append(
            next_word
        )

    return " ".join(words)

# ==========================================
# TOP K SAMPLING
# ==========================================

def generate_topk(
        model,
        prompt,
        word2idx,
        idx2word,
        device,
        k=10,
        max_new_tokens=20):

    model.eval()

    words = prompt.lower().split()

    for _ in range(max_new_tokens):

        token_ids = [

            word2idx.get(
                word,
                word2idx["<UNK>"]
            )

            for word in words

        ]

        token_ids = token_ids[-30:]

        padding = [0] * (
            30 - len(token_ids)
        )

        token_ids = padding + token_ids

        input_tensor = torch.tensor(
            [token_ids]
        ).to(device)

        with torch.no_grad():

            outputs = model(
                input_tensor
            )

        logits = outputs[:, -1, :]

        topk_logits, topk_indices = \
            torch.topk(
                logits,
                k
            )

        probs = torch.softmax(
            topk_logits,
            dim=-1
        )

        selected = torch.multinomial(
            probs,
            1
        )

        next_token = \
            topk_indices[0][selected]

        next_word = idx2word.get(
            next_token.item(),
            "<UNK>"
        )

        words.append(
            next_word
        )

    return " ".join(words)