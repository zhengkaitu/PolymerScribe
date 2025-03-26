import torch
import torch.nn as nn
import torch.nn.functional as F
from onmt.utils.misc import sequence_mask
from typing import Any, Dict, Tuple
from .tokenizer import PAD_ID, MASK, MASK_ID
from .utils import to_device


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        # assuming output is raw logits
        # convert to log_probs
        log_probs = F.log_softmax(output, dim=-1)

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        # reduction mean or sum?
        return F.kl_div(log_probs, model_prob, reduction='batchmean')


class SequenceLoss(nn.Module):

    def __init__(self, label_smoothing, vocab_size, ignore_index=-100, ignore_indices=[]):
        super(SequenceLoss, self).__init__()
        if ignore_indices:
            ignore_index = ignore_indices[0]
        self.ignore_index = ignore_index
        self.ignore_indices = ignore_indices
        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        else:
            self.criterion = LabelSmoothingLoss(label_smoothing, vocab_size, ignore_index)

    def forward(self, output, target):
        """
        :param output: [batch, len, vocab]
        :param target: [batch, len]
        :return:
        """
        batch_size, max_len, vocab_size = output.size()
        output = output.reshape(-1, vocab_size)
        target = target.reshape(-1)
        for idx in self.ignore_indices:
            if idx != self.ignore_index:
                target.masked_fill_((target == idx), self.ignore_index)
        loss = self.criterion(output, target)

        return loss


class GraphLoss(nn.Module):

    def __init__(self, num_edge_type):
        super().__init__()
        weight = torch.ones(num_edge_type) * 10
        weight[0] = 1
        self.criterion = nn.CrossEntropyLoss(weight, ignore_index=-100)

    def forward(self, outputs, targets):
        results = {}
        if 'edges' in outputs:
            pred = outputs['edges']
            max_len = pred.size(-1)
            target = targets['edges'][:, :max_len, :max_len]
            results['edges'] = self.criterion(pred, target)

        return results


class Criterion(nn.Module):

    def __init__(self, args, tokenizer):
        super(Criterion, self).__init__()
        criterion = {}
        for format_ in args.formats:
            if format_ == 'edges':
                criterion['edges'] = GraphLoss(num_edge_type=7)
            else:
                if MASK in tokenizer[format_].stoi:
                    ignore_indices = [PAD_ID, MASK_ID]
                else:
                    ignore_indices = []
                criterion[format_] = SequenceLoss(
                    args.label_smoothing,
                    len(tokenizer[format_]),
                    ignore_index=PAD_ID,
                    ignore_indices=ignore_indices
                )
        self.criterion = nn.ModuleDict(criterion)

    @staticmethod
    def get_seq_acc(
        results: Dict[str, Any],
        refs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, targets, dec_out = results["chartok_coords"]
        indices, indices_lengths = refs["full_atom_indices"]

        # print(f"predictions: {predictions}")
        # print(f"targets: {[t for t in targets]}")
        # print(f"atom_indices_targets: {atom_indices_targets}")

        # aliasing to respect the original gathering logic
        # pop last, and then insert 0 at the start
        # NOTE that this is different from the positions of the node embeddings!
        # also it still keeps the index corresponding to EOS, should be fine if fed with the lengths though.
        # indices = atom_indices[:, :-1]
        # m = nn.ConstantPad1d((1, 0), 0)
        # indices = m(indices)

        predicted_ids = torch.argmax(logits, dim=-1)        # (b, t, v) -> (b, t)
        target_mask = (targets != PAD_ID).long()
        seq_accs = (predicted_ids == targets).float()
        seq_accs = seq_accs * target_mask
        seq_acc = seq_accs.sum() / target_mask.sum()

        b, t, v = logits.size()
        # print(f"logits size: {logits.size()}")

        batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
        indices = indices.view(-1)
        # print(f"atom_indices: {atom_indices}, shape: {atom_indices.size()}")
        # print(f"indices: {indices}, shape: {indices.size()}")
        # print(f"batch_id: {batch_id}, shape: {batch_id.size()}")
        gathered_predicted_ids = predicted_ids[batch_id, indices].view(b, -1)
        gathered_targets = targets[batch_id, indices].view(b, -1)
        # b, l, dim = hidden.size()

        # print(f"gathered_predicted_ids: {gathered_predicted_ids}")
        # print(f"gathered_targets size: {gathered_targets.size()}")
        # print(f"atom_indices_lengths: {atom_indices_lengths}")
        # print(f"gathered_targets: {gathered_targets}")

        gathered_target_mask = sequence_mask(
            indices_lengths.squeeze(),
            torch.max(indices_lengths)
        )
        gathered_target_mask = to_device(gathered_target_mask, gathered_predicted_ids.device)
        # print(f"gathered_target_mask: {gathered_target_mask}")

        seq_accs_token_only = (gathered_predicted_ids == gathered_targets).float()
        seq_accs_token_only = seq_accs_token_only * gathered_target_mask
        seq_acc_token_only = seq_accs_token_only.sum() / gathered_target_mask.sum()

        return seq_acc, seq_acc_token_only

    def forward(
        self,
        results,
        refs
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        losses = {}
        for format_ in results:
            predictions, targets, *_ = results[format_]
            loss_ = self.criterion[format_](predictions, targets)
            if type(loss_) is dict:
                losses.update(loss_)
            else:
                if loss_.numel() > 1:
                    loss_ = loss_.mean()
                losses[format_] = loss_

        seq_acc, seq_acc_token_only = self.get_seq_acc(results, refs)
        # edge_tp = self.get_edge_tp(results, refs)
        edge_tp = seq_acc

        metrics = {
            "seq_acc": seq_acc,
            "seq_acc_token_only": seq_acc_token_only,
            "edge_tp": edge_tp
        }

        return losses, metrics
