import argparse
from save_check_point import save_sh_n_codes

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type = str, default = "/mnt/cephfs/home/alvin/yishen/VIGA/checkpoint/source-VIGA_moreGPU")
opt = parser.parse_args()
opt1 = vars(opt)
save_sh_n_codes(opt1, ignore_dir = [])
