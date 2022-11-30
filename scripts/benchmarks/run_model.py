import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.global_config import global_info_share_num
import torch
import argparse
import json
from datetime import datetime as dt

from rdkit import RDLogger
from seq_graph_retro.molgraph.mol_features import ATOM_FDIM
from seq_graph_retro.molgraph.mol_features import BOND_FDIM, BINARY_FDIM
from seq_graph_retro.models.model_builder import build_model, MODEL_ATTRS
from seq_graph_retro.models import Trainer
from seq_graph_retro.utils import str2bool
import wandb
import yaml

DATA_DIR = os.path.join('../..','datasets')
OUT_DIR = os.path.join('../..','experiments')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_SHARDS = global_info_share_num

def get_model_dir(model_name):
    MODEL_DIRS = {
        'single_edit': 'bond_edits',
        'multi_edit': 'bond_edits_seq',
        'single_shared': 'bond_edits',
        'lg_classifier': 'lg_classifier',
        'lg_ind': 'lg_classifier'
    }

    return MODEL_DIRS.get(model_name)

def run_model(config):
    torch.set_default_dtype(torch.float32)


    model = build_model(config, device=DEVICE)
    # torch.manual_seed(0) #尝试设置随机种子
    print(f"Converting model to device: {DEVICE}")
    sys.stdout.flush()
    model.to(DEVICE)

    print("Param Count: ", sum([x.nelement() for x in model.parameters()]) / 10 ** 6, "M")
    print()
    sys.stdout.flush()

    _, train_dataset_class, eval_dataset_class, use_labels = MODEL_ATTRS.get(config['model'])
    model_dir_name = get_model_dir(config['model'])

    train_dir = os.path.join(config['data_dir'], 'train', model_dir_name)
    eval_dir = os.path.join(config['data_dir'], 'eval')

    train_dataset = train_dataset_class(data_dir=train_dir)

    eval_dataset = eval_dataset_class(data_dir=eval_dir,data_file='info',num_shards=global_info_share_num)


    train_data = train_dataset.create_loader(batch_size=1,shuffle=False,num_workers=config['num_workers'])
    eval_data = eval_dataset.create_loader(batch_size=32,shuffle=False,num_workers=config['num_workers'])

    date_and_time = dt.now().strftime("%d-%m-%Y--%H-%M-%S")

    trainer = Trainer(model=model, print_every=config['print_every'],
                      eval_every=config['eval_every'])
    trainer.build_optimizer(learning_rate=config['lr'],finetune_encoder=False)
    trainer.build_scheduler(type=config['scheduler_type'], anneal_rate=config['anneal_rate'],
                            patience=config['patience'], thresh=config['metric_thresh'])

    trainer.train_epochs(train_data, eval_data, config['epochs'],
                         **{"accum_every": config.get('accum_every', None),
                            "clip_norm": config['clip_norm']})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--out_dir', default=OUT_DIR)
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--num_workers', default=6, type=int)
    args = parser.parse_args()

    wandb.init(project='my_multiedit',dir=args.out_dir,config=args.config_file)
    config = wandb.config
    tmp_dict = vars(args)
    for key, value in tmp_dict.items():
        config[key] = value

    print(config)

    run_model(config)



    pass



if __name__ == '__main__':
    main()



