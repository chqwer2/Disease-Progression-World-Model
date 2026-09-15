import torch
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


# config.margin_list = (1.0, 0.0, 0.4)
class CombinedMarginLoss(torch.nn.Module):
    def __init__(self,
                 s,
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits


import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelDifference(nn.Module):
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)


class FeatureSimilarity(nn.Module):
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            # For every pair (i, j), you're computing:
            return - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)


class Origin_RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(Origin_RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        featues_a = features[:, 0]  # [bs, feat_dim]
        featues_b = features[:, 1]  # [bs, feat_dim]

        features = torch.cat([featues_a, featues_b], dim=0)  # [2bs, feat_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            pos_logits = logits[:, k]  # 2bs
            pos_label_diffs = label_diffs[:, k]  # 2bs
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()

        return loss


class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels, ids):
        # [S1, S2, S3]

        # Compute pairwise similarity and label differences
        logits = self.feature_sim_fn(features).div(self.t)
        label_diffs = self.label_diff_fn(labels)

        # Intra-patient mask: [n, n]
        N = logits.shape[0]
        # eye = torch.eye(N, device=features.device).bool()

        ids_i = ids.view(-1, 1)
        ids_j = ids.view(1, -1)
        same_patient_mask = (ids_i == ids_j).float()  # 1 if same patient, else 0, matrix

        # Remove self-pairs (diagonal)
        eye = torch.eye(N, device=features.device)
        same_patient_mask = same_patient_mask * (1 - eye)
        same_patient_mask = same_patient_mask.bool()  # Convert to boolean mask

        # Compute RnC loss
        total_loss = 0.0
        count = 0

        print("labels=", labels)
        print("logits= ", logits)

        for k in range(N):
            mask_i = same_patient_mask[k]  # [N], valid comparisons for sample i
            if mask_i.sum() == 0:
                continue

            sim_ij = logits[k][mask_i]  # similarities to other timepoints
            label_diff_ij = label_diffs[k][mask_i]

            scaled_sim = -label_diff_ij * sim_ij
            log_prob = scaled_sim - torch.logsumexp(scaled_sim, dim=0)
            loss_i = -log_prob.mean()

            total_loss += loss_i
            count += 1

        return total_loss / count


def monotonicity_triplet_loss(features, labels, ids=None, margin=0.1, mode='magnitude'):
    """
    ArcRank temporal-ranking term on the SVD singular values ``features`` (Sigma).

    The batch is stacked as three time-ordered visits [t1 | t2 | t3] of the SAME
    patients along dim 0. For each patient whose ages satisfy a1 <= a2 <= a3:

    mode='magnitude'  (paper-faithful, default) --
        Enforce the latent magnitude  ||z|| = sum(Sigma)  to grow monotonically:
            relu(margin - (M2 - M1)) + relu(margin - (M3 - M2)),   M = sum of Sigma
        i.e. Sigma_{t+1} - Sigma_t >= margin. This is the paper's ranking loss
            L_rank = sum_{i<j} max(0, m - (Sigma_j - Sigma_i)).

    mode='displacement'  (alternative, kept for ablation) --
        Enforce later visits to drift farther from baseline: relu(d(S1,S2) - d(S1,S3) + margin).

    Returns a differentiable 0 when the batch contains no age-ordered triple, so
    training is not interrupted.
    """
    N = features.shape[0]
    B = N // 3
    feat = features.reshape(N, -1)

    a1, a2, a3 = labels[:B], labels[B:2 * B], labels[2 * B:]
    ordered = (a1 <= a2) & (a2 <= a3)
    if ordered.dim() > 1:                       # labels may arrive as (B,1)
        ordered = ordered.reshape(B, -1).all(dim=1)

    if mode == 'magnitude':
        mag = feat.sum(dim=1)                   # ||z|| proxy (nuclear norm; Sigma >= 0)
        m1, m2, m3 = mag[:B], mag[B:2 * B], mag[2 * B:]
        term = F.relu(margin - (m2 - m1)) + F.relu(margin - (m3 - m2))
    elif mode == 'displacement':
        s1, s2, s3 = feat[:B], feat[B:2 * B], feat[2 * B:]
        d12 = F.pairwise_distance(s1, s2, p=2)
        d13 = F.pairwise_distance(s1, s3, p=2)
        term = F.relu(d12 - d13 + margin)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    ordered = ordered.to(term.dtype)
    denom = ordered.sum().clamp_min(1.0)        # never divide by zero
    return (term * ordered).sum() / denom


# closer the smaller loss
class TripletRnCLoss(nn.Module):
    def __init__(self, margin=0.2, feature_sim='l2'):
        super().__init__()
        self.margin = margin
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def pairwise_distance(self, x, y):
        if self.feature_sim_fn.mode == 'cosine':
            return 1.0 - F.cosine_similarity(x, y)
        elif self.feature_sim_fn.mode == 'l2':
            return F.pairwise_distance(x, y, p=2)
        else:
            raise ValueError("Unsupported distance metric.")

    def forward(self, features, labels, ids):
        # features: [N, D], labels: [N, 1], ids: [N, 1]

        N = features.shape[0]
        total_loss = 0.0
        count = 0

        for k in range(N):
            anchor = features[k]
            anchor_label = labels[k]
            anchor_id = ids[k]

            # All other timepoints from the same patient (excluding self)
            same_patient = (ids == anchor_id) & (torch.arange(N, device=ids.device) != k)
            if same_patient.sum() < 2:
                continue  # Need at least two others to pick both positive & negative

            label_diffs = torch.abs(labels - anchor_label)
            valid_indices = torch.where(same_patient)[0]
            sorted_by_diff = valid_indices[torch.argsort(label_diffs[valid_indices])]

            # Positive: smallest label difference
            pos_idx = sorted_by_diff[0]
            # Negative: largest label difference
            neg_idx = sorted_by_diff[-1]

            positive = features[pos_idx]
            negative = features[neg_idx]

            d_ap = self.pairwise_distance(anchor.unsqueeze(0), positive.unsqueeze(0))  # [1]
            d_an = self.pairwise_distance(anchor.unsqueeze(0), negative.unsqueeze(0))  # [1]

            triplet_loss = F.relu(d_ap - d_an + self.margin)
            total_loss += triplet_loss
            count += 1

        if count == 0:
            raise ValueError("TripletRnCLoss: No valid triplets found.")

        return total_loss / count


# represents a directional embedding for a sample
class AngleLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        # temperature is a learnable parameter in original CLIP
        self.logit_scale = nn.Parameter(torch.tensor(1 / temperature).log())

    def forward(self, x1, x2):
        # Normalize the features
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)

        # Compute cosine similarity scaled by temperature
        logit_scale = self.logit_scale.exp()
        logits_per_image = torch.matmul(x1, x2.t()) * logit_scale
        logits_per_text = logits_per_image.t()

        # Ground truth: diagonal is the correct pair
        batch_size = x1.size(0)
        labels = torch.arange(batch_size, device=x1.device)

        # Cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)

        # Final symmetric loss
        return (loss_i2t + loss_t2i) / 2


class DistCrossEntropyFunc(torch.autograd.Function):
    """
    CrossEntropy loss is calculated in parallel, allreduce denominator into single gpu and calculate softmax.
    Implemented of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, label: torch.Tensor):
        """ """
        batch_size = logits.size(0)
        # for numerical stability
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(max_logits, distributed.ReduceOp.MAX)
        logits.sub_(max_logits)
        logits.exp_()
        sum_logits_exp = torch.sum(logits, dim=1, keepdim=True)
        # local to global
        distributed.all_reduce(sum_logits_exp, distributed.ReduceOp.SUM)
        logits.div_(sum_logits_exp)
        index = torch.where(label != -1)[0]
        # loss
        loss = torch.zeros(batch_size, 1, device=logits.device)
        loss[index] = logits[index].gather(1, label[index])
        distributed.all_reduce(loss, distributed.ReduceOp.SUM)
        ctx.save_for_backward(index, logits, label)
        return loss.clamp_min_(1e-30).log_().mean() * (-1)

    @staticmethod
    def backward(ctx, loss_gradient):
        """
        Args:
            loss_grad (torch.Tensor): gradient backward by last layer
        Returns:
            gradients for each input in forward function
            `None` gradients for one-hot label
        """
        (
            index,
            logits,
            label,
        ) = ctx.saved_tensors
        batch_size = logits.size(0)
        one_hot = torch.zeros(
            size=[index.size(0), logits.size(1)], device=logits.device
        )
        one_hot.scatter_(1, label[index], 1)
        logits[index] -= one_hot
        logits.div_(batch_size)
        return logits * loss_gradient.item(), None


class DistCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(DistCrossEntropy, self).__init__()

    def forward(self, logit_part, label_part):
        return DistCrossEntropyFunc.apply(logit_part, label_part)

