import torch
from rdkit import Chem
import argparse
import numpy as np
import pandas as pd
import joblib
import os
import sys
import copy
from typing import Any

from configs.global_config import global_info_share_num
from seq_graph_retro.utils.parse import get_reaction_info,extract_leaving_groups
from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol
from seq_graph_retro.molgraph import BondEditsRxn, RxnElement, MultiElement
from seq_graph_retro.molgraph.vocab import Vocab
from seq_graph_retro.data.collate_fns import (pack_graph_feats, prepare_lg_labels,
                                              tensorize_bond_graphs)
from seq_graph_retro.utils import str2bool

DATA_DIR = '../datasets'

def parse_info(df, args):
    info_all = []
    counter = []
    mol_list = []

    legal_cnt = 0

    save_dir = os.path.join(args.data_dir,f'{args.mode}')
    os.makedirs(save_dir,exist_ok=True)
    save_file = 'info'

    for idx in range(len(df)):
        element = df.iloc[idx]
        rxn_smi = element['reactants>reagents>production']
        MainId = element['id']
        Order = idx
        try:
            reaction_info = get_reaction_info(rxn_smi, kekulize=True,
                                              use_h_labels=True,mainid=MainId,order=Order)
        except Exception as e:
            # print(f"Failed to extract reaction info. Skipping reaction {idx}")
            # print("due to: ", e)
            # sys.stdout.flush()
            # print(rxn_smi, file=ex_rxn_f)
            continue

        r, p = rxn_smi.split(">>")
        products = get_mol(p)

        if (products.GetNumBonds() <= 1):
            print(f'Product has 0 or 1 bonds, Skipping reaction {idx}')
            print()
            sys.stdout.flush()
            continue

        if (products is None) or (products.GetNumAtoms() <= 1):
            print(f"Product has 0 or 1 atoms, Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        reactants = get_mol(r)
        if (reactants is None) or (reactants.GetNumAtoms() <= 1):
            print(f"Reactant has 0 or 1 atoms, Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        fragments = apply_edits_to_mol(Chem.Mol(products), reaction_info.core_edits)
        counter.append(len(reaction_info.core_edits))

        if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
            print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        frag_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(fragments)).mols)
        reac_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(reactants)).mols)
        mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))
        info_all.append(reaction_info)
        legal_cnt = legal_cnt + 1
        if(idx % args.print_every == 0 ) and idx:
            print(f"{idx}/{len(df)} {args.mode} reactions processed.", f'{legal_cnt} legal reactions!')
            sys.stdout.flush()

    print(f"All {args.mode} reactions complete.")
    sys.stdout.flush()

    n_shards = global_info_share_num
    indices_shards = np.array_split(np.arange(len(info_all)), n_shards)

    info_file = os.path.join(save_dir, save_file)
    for shard_num, indices_per_shard in enumerate(indices_shards):
        info_shard = []
        frag_shard = []

        for index in indices_per_shard:
            info_shard.append(info_all[index])

        info_file_shard = info_file + f"-shard-{shard_num}"
        joblib.dump(info_shard, info_file_shard, compress=3)

    from collections import Counter
    print(Counter(counter))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--print_every', default=1000, type=int)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    filename = f'{args.mode}_after_1_c.csv'

    data_df = pd.read_csv(os.path.join(args.data_dir, filename))

    parse_info(data_df, args)


if __name__ == '__main__':
    main()