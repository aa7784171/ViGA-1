import os
import yaml
import torch
import random
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm

from src.dataset.dataset import prepare_data
from src.utils.utils import load_config, n_params, get_now
from src.utils.vl_utils import GloVe
from eval import Evaluator
from src.model.model import Model
from save_check_point import save_sh_n_codes
from tensorboardX import SummaryWriter



SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(config, args):
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config['device_ids']
    
    # save log
    exp_folder_path = os.path.join(config["checkpoint_path"])
    # exp_folder_path = os.path.join(config["exp_dir"], "{}_{}_{}".format(args.task,args.name, get_now()))
    # Path(exp_folder_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(exp_folder_path, "config.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    #tensorboard 
    writer = SummaryWriter(config['checkpoint_path'] + "/tensorboard")
    
    # prepare data
    dataset_name = config["dataset_name"]
    name = config["name"]
    data = prepare_data(config, dataset_name)
    train_dl = data["train_dl"]
    # valid_dl = data["valid_dl"]
    test_dl = data["test_dl"]

    vocab = data["vocab"]
    glove = GloVe(glove_path=config["model"]["glove_path"])
    model = Model(config, vocab, glove)
    model = torch.nn.DataParallel(model).cuda()
    
    #create optimizer, scheduler
    lr = config["train"]["init_lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=3
        )
    
    print("Model has {} parameters.\n".format(n_params(model)))


    test_evaluator = Evaluator()
    log_file_path = os.path.join(exp_folder_path, "train.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("{}\n".format(config[dataset_name]["feature_dir"]))
        log_file.write("Model has {} parameters.\n".format(n_params(model)))
        log_file.flush()
        for epoch in range(1, config[dataset_name]["epoch"] + 1):
            max1_train = 0
            max5_train = 0
            min1_train = 0
            min5_train = 0
            total_train = 0
            for i, batch in tqdm(
                enumerate(train_dl), total=len(train_dl),
                desc="Training epoch {} with lr {}".format(epoch, optimizer.param_groups[0]["lr"])
            ):
                model.train()
                loss, attn_weights = model(batch,"train")
                loss = loss.mean()
                print(loss.item())
                #seed_acc
                total_train += len(batch['idx'])
                batch['start_frame'] = batch['start_frame'].cuda()
                batch['end_frame'] =  batch['end_frame'].cuda()

                for i in range(len(batch['idx'])):
                    if batch['start_frame'][i] <= (torch.topk(attn_weights,1).indices+1)[i] <= batch['end_frame'][i]:
                        max1_train += 1
                    for j in (torch.topk(attn_weights,5).indices+1)[i]:
                        if batch['start_frame'][i] <= j <= batch['end_frame'][i]:
                            max5_train += 1
                            break    
                    
                    if not batch['start_frame'][i] <= (torch.topk(attn_weights,1,largest=False).indices+1)[i] <= batch['end_frame'][i]:
                        min1_train += 1
                    for j in (torch.topk(attn_weights,5,largest=False).indices+1)[i]:
                        if not batch['start_frame'][i] <= j <= batch['end_frame'][i]:
                            min5_train += 1
                            break            
                #optimizer
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["clip_norm"])
                optimizer.step()
                
            print("traing_seedgt_max_top1:{}%".format(max1_train/total_train*100) + "\n")
            print("traing_seedgt_max_top5:{}%".format(max5_train/total_train*100) + "\n")
            print("traing_seedgt_min_top1:{}%".format(min1_train/total_train*100) + "\n")
            print("traing_seedgt_min_top5:{}%".format(min5_train/total_train*100) + "\n")
            log_file.write("\n==== epoch {} ====\n".format(epoch))
            log_file.write("        ## training-seed_acc ##\n")
            log_file.write("seedgt_max_top1:{}%".format(max1_train/total_train*100) + "\n")
            log_file.write("seedgt_max_top5:{}%".format(max5_train/total_train*100) + "\n")
            log_file.write("seedgt_min_top1:{}%".format(min1_train/total_train*100) + "\n")
            log_file.write("seedgt_min_top5:{}%".format(min5_train/total_train*100) + "\n")
            with torch.no_grad():
                test_loss, total_eval, max1_eval, max5_eval, min1_eval, min5_eval = test_evaluator.eval_dataloader(model, test_dl, epoch)
                scheduler.step(test_loss)
                
                writer.add_scalars("loss",{'train_loss':loss.item()},epoch)
                writer.add_scalars("loss",{'test_loss':test_loss},epoch)
                writer.close()
                
            print("testing_seedgt_max_top1:{}%".format(max1_eval/total_eval*100) + "\n")
            print("testing_seedgt_max_top5:{}%".format(max5_eval/total_eval*100) + "\n")
            print("testing_seedgt_min_top1:{}%".format(min1_eval/total_eval*100) + "\n")
            print("testing_seedgt_min_top5:{}%".format(min5_eval/total_eval*100) + "\n")
            # log_file.write("\n==== epoch {} ====\n".format(epoch))
            log_file.write("        ## testing-seed_acc ##\n")
            log_file.write("seedgt_max_top1:{}%".format(max1_eval/total_eval*100) + "\n")
            log_file.write("seedgt_max_top5:{}%".format(max5_eval/total_eval*100) + "\n")
            log_file.write("seedgt_min_top1:{}%".format(min1_eval/total_eval*100) + "\n")
            log_file.write("seedgt_min_top5:{}%".format(min5_eval/total_eval*100) + "\n")
           
            # log_file.write("\n==== epoch {} ====\n".format(epoch))
            log_file.write("        ## test ##\n")
            log_file.write(test_evaluator.report_current() + "\n")
            log_file.write(test_evaluator.report_best() + "\n")
            log_file.flush()

            # save best
            if epoch == test_evaluator.best_epoch:
                torch.save(model.state_dict(), os.path.join(exp_folder_path, "model_{}.pt".format("best")))
                print("== Checkpoint ({}) is saved to {}".format("best", exp_folder_path))
            print("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--task", help="Dataset name in {activitynetcaptions, charadessta, tacos}.", required=True)
    parser.add_argument("--name", type=str, default="base")
    parser.add_argument('--checkpoint_path', type = str, default = "/mnt/cephfs/home/alvin/yishen/CFVG/try/model_checkpoints/source_VIGA/charades_c3d")
    parser.add_argument("--mode", help="train or debug", default="train")
    parser.add_argument('--device_ids',type=str, default = "0")

    

    
    args = parser.parse_args()
    #是否保存代码
    if args.mode == "train":
        opt1 = vars(args)
        save_sh_n_codes(opt1, ignore_dir = ["data"])
    
    config = load_config("src/config.yaml")
    config["dataset_name"] = args.task
    config["name"] = args.name
    config["checkpoint_path"] = args.checkpoint_path
    config["device_ids"] = args.device_ids
    
    train(config,args)
