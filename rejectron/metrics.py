from typing import Any, Dict, Union

import torch
from torchmetrics import Metric, Accuracy


def itemize(dict_: Dict[str, Union[torch.Tensor, float]]) -> Dict[str, float]:
    return {k: v if not isinstance(v, torch.Tensor) else v.item() for k, v in dict_.items()}


class RejectronMetric(Metric):
    def __init__(self, beta=1, val_metric=False, domain_metric=False):
        super().__init__()

        # track number of samples in p that h and c agree on
        self.p_agree = 0
        # track number of samples in q that h and c disagree on
        self.q_disagree = 0

        # of samples that h and c agree on, tracks accuracy in both p and q
        self.acc_p = Accuracy()
        self.acc_q = Accuracy()

        self.acc_p_all = Accuracy()
        self.acc_q_all = Accuracy()

        self.p_seen = 0
        self.q_seen = 0

        self.beta = beta
        self.val = val_metric
        self.domain = domain_metric

    def update(self, labels: torch.Tensor, c_logits: torch.Tensor, h_pred: torch.Tensor):
        """
        Args:
            labels: positive labels for elements of p, negative labels for elements of q
            c_logits:
            h_pred:

        Returns:

        """
        c_pred = c_logits.argmax(dim=1)
        # if domain is true we know h_pred == labels
        p_label_indices = labels >= 0 if not self.domain else labels == 1
        q_label_indices = ~p_label_indices

        num_p = p_label_indices.sum()
        self.p_seen += num_p
        num_q = len(labels) - num_p
        self.q_seen += num_q

        if (p_label_indices == True).all():
            # every sample is from p
            q_disagree = 0
            p_agree = (c_pred == h_pred)
            self.acc_p_all.update(c_pred, labels)
            self.acc_p.update(c_pred[p_agree], labels[p_agree])
            p_agree = p_agree.sum().item()

        elif (q_label_indices == True).all():
            # every sample is from q
            p_agree = 0
            if self.domain:
                q_disagree = (c_pred == h_pred)
                # this is meaningless here, but we log it just for consistency
                self.acc_q.update(c_pred[q_disagree], labels[q_disagree])
                self.acc_q_all.update(c_pred, labels)
            else:
                q_disagree = (c_pred != h_pred)
                self.acc_q.update(c_pred[~q_disagree], -labels[~q_disagree] - 1)
                self.acc_q_all.update(c_pred, -labels - 1)
            q_disagree = q_disagree.sum().item()

        else:
            c_p = c_pred[p_label_indices]
            c_q = c_pred[q_label_indices]

            p_labels = labels[p_label_indices]
            q_labels = labels[q_label_indices]

            self.acc_p_all.update(c_p, p_labels)
            if self.domain:
                self.acc_q_all.update(c_q, q_labels)
            else:
                self.acc_q_all.update(c_q, -q_labels - 1)

            p_mask = (c_p == h_pred[p_label_indices])
            if not self.domain:
                q_mask = (c_q == h_pred[q_label_indices])
            else:
                q_mask = (c_q != h_pred[q_label_indices])

            # updates accuracy metrics where h and c agree
            if (p_agree := p_mask.sum().item()) > 0:
                self.acc_p.update(c_p[p_mask], p_labels[p_mask])

            if (~q_mask).sum().item() > 0:
                if not self.domain:
                    self.acc_q.update(c_q[~q_mask], -q_labels[~q_mask] - 1)
                else:
                    self.acc_q.update(c_q[~q_mask], q_labels[~q_mask])

            q_disagree = q_mask.sum().item()

        self.p_agree += p_agree
        self.q_disagree += q_disagree

        # return dict(p_agree=p_agree / num_q, q_disagree=q_disagree / num_q, p_acc=acc_p, q_acc=acc_q)

    @staticmethod
    def acc_or_1(acc_metric: Accuracy):
        if acc_metric.mode is None:
            return 1
        return acc_metric.compute()

    def compute(self) -> Any:
        if self.val:
            return itemize(dict(
                val_agree=self.p_agree / self.p_seen,
                val_acc=self.acc_p.compute(),
                val_acc_all=self.acc_p_all.compute()
            ))
        test_acc = self.acc_or_1(self.acc_q)
        test_acc_all = self.acc_or_1(self.acc_q_all)
        return itemize(
            dict(
                train_agree=self.p_agree / self.p_seen,
                test_reject=self.q_disagree / self.q_seen,
                train_acc=self.acc_p.compute(),
                train_acc_all=self.acc_p_all.compute(),
                test_acc=test_acc,
                test_acc_all=test_acc_all,
                # overall P/Q score:
                #   a score of 100 is achieved by agreeing on everything in P and disagreeing on everything in Q
                p_q_score=(100 * self.p_agree / self.p_seen).round() + self.q_disagree / self.q_seen
            )
        )

    def reset(self) -> None:
        self.p_agree = 0
        self.q_disagree = 0

        self.acc_p.reset()
        self.acc_q.reset()
        self.acc_q_all.reset()
        self.acc_p_all.reset()

        self.p_seen = 0
        self.q_seen = 0
