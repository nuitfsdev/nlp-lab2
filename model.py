import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


MAX_LENGTH = 128
PAD_TOKEN_ID = 0
SEP_TOKEN_ID = 102


def tokenizes(examples, tokenizer):
    """Tokenize inputs in the format used by the supplied test.py."""
    if isinstance(examples, dict):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    # test.py joins the pair using this marker. Split it back into a real
    # sentence pair so BERT creates [CLS] premise [SEP] hypothesis [SEP].
    if isinstance(examples, str) and " [CLS] " in examples:
        premise, hypothesis = examples.split(" [CLS] ", maxsplit=1)
        return tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    return tokenizer(
        examples,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def collate_fn(batch):
    """Create exactly the arguments accepted by NLI.forward.

    The supplied test.py only keeps input_ids and labels, so the attention
    mask is rebuilt here from MiniLM/BERT's padding token ID (zero).
    """
    input_ids = torch.tensor(
        [sample["input_ids"] for sample in batch], dtype=torch.long
    )
    attention_mask = input_ids.ne(PAD_TOKEN_ID).long()
    positions = torch.arange(input_ids.size(1)).unsqueeze(0)
    first_sep_positions = input_ids.eq(SEP_TOKEN_ID).long().argmax(dim=1).unsqueeze(1)
    token_type_ids = ((positions > first_sep_positions) & attention_mask.bool()).long()
    labels = torch.tensor(
        [int(sample["labels"]) for sample in batch], dtype=torch.long
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


class NLIConfig(PretrainedConfig):
    model_type = "nli_minilm"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=384,
        nclass=3,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=1536,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        pad_token_id=PAD_TOKEN_ID,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass
        self.num_labels = nclass
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob


class NLI(PreTrainedModel):
    config_class = NLIConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        encoder_config = BertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            type_vocab_size=config.type_vocab_size,
            pad_token_id=config.pad_token_id,
        )
        self.encoder = BertModel(encoder_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.nclass)
        self.loss_fct = nn.CrossEntropyLoss()
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        token_type_ids=None,
        **kwargs,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
