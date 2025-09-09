import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig

class DualMaskedBertConfig(BertConfig):
    """拡張設定クラス（現状追加項目なし）"""
    model_type = "dualmasked_bert"

class DualMaskedBertForPreTraining(BertPreTrainedModel):
    """文脈ベクトル予測 + トークン再現の 2 ヘッドモデル"""
    config_class = DualMaskedBertConfig

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)

        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        # ベクトル再現ヘッド (hidden -> hidden)
        self.vector_pred = nn.Linear(hidden_size, hidden_size)
        # トークン再現ヘッド (hidden -> vocab)
        self.token_pred = nn.Linear(hidden_size, vocab_size, bias=False)
        self.token_pred.weight = self.bert.embeddings.word_embeddings.weight  # weight tying

        self.init_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        labels_token: torch.LongTensor | None = None,
        labels_vector: torch.Tensor | None = None,
    ):
        """入力:
        - input_ids: マスク済み入力
        - labels_token: マスクトークン id (非マスク位置は -100)
        - labels_vector: マスク位置の元 hidden ベクトル (shape: [batch, seq_len, hidden])、非マスクは 0
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state  # (B, L, H)

        # 予測
        pred_vectors = self.vector_pred(sequence_output)  # (B, L, H)
        logits = self.token_pred(sequence_output)  # (B, L, V)

        loss = None
        loss_token = None
        loss_vector = None
        if labels_token is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_token = loss_fct(logits.view(-1, logits.size(-1)), labels_token.view(-1))
        if labels_vector is not None:
            # MSE で計算（マスク外は zeros -> 重み0 になるよう attention_mask で除外）
            mse = nn.MSELoss(reduction="none")
            diff = mse(pred_vectors, labels_vector)  # (B,L,H)
            # mask: ラベルが 0 でない位置 (any dimension) を有効とみなす
            mask = (labels_vector.abs().sum(dim=-1) != 0).float()  # (B,L)
            loss_vector = (diff.mean(-1) * mask).sum() / (mask.sum() + 1e-8)
        if loss_token is not None and loss_vector is not None:
            loss = loss_token + loss_vector
        elif loss_token is not None:
            loss = loss_token
        elif loss_vector is not None:
            loss = loss_vector

        return {
            "loss": loss,
            "loss_token": loss_token,
            "loss_vector": loss_vector,
            "logits": logits,
            "pred_vectors": pred_vectors,
        }
