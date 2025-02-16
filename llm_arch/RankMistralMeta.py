from abc import ABC, abstractmethod
import torch
import torch.nn as nn


def build_graph_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.graph_hidden_dim, 4096)


class RankMistralMeta:
    def __init__(self, config):
        super(RankMistralMeta, self).__init__(config)
        if hasattr(config, "graph_hidden_dim"):
            self.mm_projector = build_graph_projector(config)

    def initialize_graph_modules(self, model_args, fsdp=None):
        self.config.graph_hidden_dim = getattr(model_args, "graph_hidden_dim")
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")

        pretrain_mm_mlp_adapter = getattr(model_args, "pretrain_mm_mlp_adapter", None)
        self.mm_projector = build_graph_projector(self.config)
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))


class RankMetaForCausalLLM(ABC):

    @staticmethod
    def get_model(self):
        pass

    def encode_graphs(self, graph, graph_emb):
        graph_features = self.get_model().mm_projector(graph_emb)
        return graph_features

    def prepare_inputs_labels_for_multimodal(
            self, input_ids, attention_mask, past_key_values, labels, graphs, graph_emb
    ):
        if past_key_values is not None and graphs is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                                        dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        graph_features = self.encode_graphs(graphs, graph_emb)
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # add graph features to prefix to the input embeds
            cur_new_input_embed = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            cur_graph_embeds = graph_features[batch_idx]
            cur_new_input_embed.append(cur_graph_embeds)
            cur_new_input_embed.append(self.get_model().embed_tokens(cur_input_ids))
            # prepare the prefix labels
            if labels is not None:
                cur_new_labels.append(
                    torch.full((cur_graph_embeds.shape[0],), -100, dtype=labels.dtype, device=labels.device))
                cur_new_labels.append(cur_labels)
            cur_new_input_embed = [x.to(device=self.device) for x in cur_new_input_embed]
            cur_new_input_embed = torch.cat(cur_new_input_embed, dim=0)
            new_input_embeds.append(cur_new_input_embed)
            if labels is not None:
                cur_new_labels = [x.to(device=self.device) for x in cur_new_labels]
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        if labels is not None:
            new_labels = torch.stack(new_labels, dim=0)
        if attention_mask is not None:
            new_attention_mask_pad_left = torch.full(
                (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]),
                True, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([new_attention_mask_pad_left, attention_mask], dim=1)
            assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
