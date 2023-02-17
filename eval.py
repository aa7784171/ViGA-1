import os
from collections import OrderedDict

import torch
import argparse

from tqdm import tqdm

from src.dataset.dataset import prepare_data
from src.utils.utils import load_config
from src.utils.vl_utils import GloVe
from src.model.model import Model


class Evaluator(object):
    def __init__(self):
        # self.recall_at = [1, 5]
        self.recall_at = [1]
        self.iou_threshold = [0.3, 0.5, 0.7]

        self.res = None
        self.epoch = None
        self.score = None

        self.best_res = None
        self.best_epoch = None
        self.best_score = None

    def _res_to_string(self, res):
        return "\n".join([" | ".join([
            "R@{} IoU {}: {:.2f}".format(recall_at, thres, res[recall_at][thres] * 100) for thres in self.iou_threshold
        ] + ["mIoU: {:.2f}".format(res[recall_at]["miou"] * 100)]) for recall_at in self.recall_at])

    @staticmethod
    def _res_to_score(res):
        """ Method to judge best epoch, currently just sum all metrics up.
        """
        return sum([sum([v for _, v in d.items()]) for _, d in res.items()])

    def _update(self, preds, gts, epoch=0):
        """
        Args:
            preds: (N, topk, 2)
            gts: (N, 2)
        """
        N, topk, _ = preds.shape
        gts = gts.unsqueeze(1)
        intersection = torch.clamp(torch.min(preds[:, :, 1], gts[:, :, 1]) - torch.max(preds[:, :, 0], gts[:, :, 0]), min=0.0)
        union = torch.clamp(torch.max(preds[:, :, 1], gts[:, :, 1]) - torch.min(preds[:, :, 0], gts[:, :, 0]), min=0.0)
        iou = intersection / union
        iou[torch.logical_or(intersection == 0.0, union == 0.0)] = 0.0  # (B, topk)
        miou = torch.mean(iou, dim=0)

        res = OrderedDict()
        for recall_at in self.recall_at:
            d = OrderedDict()
            temp = iou[:, :recall_at]
            for thres in self.iou_threshold:
                count = torch.sum(torch.any(temp >= thres, dim=1).to(torch.long)).item()
                d[thres] = round(count / N, 4)
            d["miou"] = miou[recall_at - 1].item()
            res[recall_at] = d

        # update result and best result
        self.res = res
        self.epoch = epoch
        self.score = self._res_to_score(res)
        if self.best_res is None or self._res_to_score(res) > self._res_to_score(self.best_res):
            self.best_res = res
            self.best_epoch = epoch
            self.best_score = self._res_to_score(res)

    def eval_dataloader(self, model, dataloader, epoch=0):
        model.eval()
        preds = []
        gts = []
        loss = 0
        max1_eval = 0
        max5_eval = 0
        min1_eval = 0
        min5_eval = 0
        total_eval = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc="Evaluating epoch {}".format(epoch)):
                pred = model(batch,"eval")
                loss_temp, attn_weights =model(batch,"train") 
                loss_temp = loss_temp.mean()
                loss += loss_temp.item()
                preds.append(pred)
                gts.append(torch.stack([batch["start_frac"], batch["end_frac"]], dim=1))
                
                #seed_acc                
                total_eval += len(batch['idx'])
                batch['start_frame'] = batch['start_frame'].cuda()
                batch['end_frame'] = batch['end_frame'].cuda()

                for i in range(len(batch['idx'])):
                    if batch['start_frame'][i] <= (torch.topk(attn_weights,1).indices+1)[i] <= batch['end_frame'][i]:
                        max1_eval += 1
                    for j in (torch.topk(attn_weights,5).indices+1)[i]:
                        if batch['start_frame'][i] <= j <= batch['end_frame'][i]:
                            max5_eval += 1
                            break    
                    
                    if not batch['start_frame'][i] <= (torch.topk(attn_weights,1,largest=False).indices+1)[i] <= batch['end_frame'][i]:
                        min1_eval += 1
                    for j in (torch.topk(attn_weights,5,largest=False).indices+1)[i]:
                        if not batch['start_frame'][i] <= j <= batch['end_frame'][i]:
                            min5_eval += 1
                            break  
                        
        loss /= len(dataloader)
        preds = torch.cat(preds, dim=0)
        gts = torch.cat(gts, dim=0).cuda()
        self._update(preds, gts, epoch)
        print(self.report_current())
        print(self.report_best())
        return loss, total_eval, max1_eval, max5_eval, min1_eval, min5_eval


    def report_current(self):
        return "This epoch {}\n{}".format(self.epoch, self._res_to_string(self.res))

    def report_best(self):
        return "Best epoch {}\n{}".format(self.best_epoch, self._res_to_string(self.best_res))


def evaluate(exp_folder_path):
    config = load_config(os.path.join(exp_folder_path, "config.yaml"))
    data = prepare_data(config, config["dataset_name"])
    test_dl = data["test_dl"]

    vocab = data["vocab"]
    glove = GloVe(glove_path=config["model"]["glove_path"])

    model = Model(config, vocab, glove)
    model.load_checkpoint(exp_folder_path, "best")
    model.eval_mode()
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.gpu_mode()
    else:
        model.cpu_mode()

    evaluator = Evaluator()
    evaluator.eval_dataloader(model, test_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Eval trained model.")
    parser.add_argument("--exp", help="Experiment folder to evaluate.", required=True)
    args = parser.parse_args()
    evaluate(args.exp)
