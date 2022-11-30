import torch
from rdkit import Chem
import argparse
import joblib
import os
import sys
import copy
from typing import Any

from configs.global_config import global_info_share_num
from seq_graph_retro.utils.parse import extract_leaving_groups
from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol
from seq_graph_retro.molgraph import BondEditsRxn, RxnElement, MultiElement
from seq_graph_retro.molgraph.vocab import Vocab
from seq_graph_retro.data.collate_fns import (pack_graph_feats, prepare_lg_labels,
        tensorize_bond_graphs)
from seq_graph_retro.utils import str2bool

DATA_DIR = '../datasets'

def process_batch(edit_graphs, mol_list, args):
    '''

    :param edit_graphs:
    :param mol_list:
    :param args:
    :return:
    '''
    assert len(edit_graphs) == len(mol_list)
    mol_attrs = ['prod_mol', 'frag_mol']
    label_attrs = ['edit_label', 'h_label']
    attributes = [graph.get_attributes(mol_attrs=mol_attrs, label_attrs=label_attrs) for graph in edit_graphs]
    prod_batch, frag_batch, edit_labels = list(zip(*attributes))

    if len(edit_labels[0]) == 1:
        edit_labels = torch.tensor(edit_labels, dtype=torch.long)
    else:
        edit_labels = [torch.tensor(edit_labels[i], dtype=torch.float) for i in range(len(edit_labels))]

    prod_inputs = pack_graph_feats(prod_batch, directed=True, use_rxn_class=False)
    frag_inputs = pack_graph_feats(frag_batch, directed=True, use_rxn_class=False)

    bond_graph_inputs = tensorize_bond_graphs(prod_batch, directed=True, use_rxn_class=False)

    return prod_inputs, edit_labels, frag_inputs, bond_graph_inputs


def parse_bond_edits_forward(args: Any, mode: str = 'train') -> None:
    '''

    :param args:
    :param mode:
    :return:
    '''

    info_all = []
    base_file = os.path.join(args.data_dir,mode,'info')
    for shard_num in range(global_info_share_num):
        shard_file = base_file + f'-shard-{shard_num}'
        info_all.extend(joblib.load(shard_file))

    bond_edits_graphs = []
    mol_list = []

    save_dir = os.path.join(args.data_dir,mode,'bond_edits')
    os.makedirs(save_dir, exist_ok=True)

    num_batches = 0
    total_examples = 0

    for idx, reaction_info in enumerate(info_all):
        rxn_smi = reaction_info.rxn_smi
        r, p = rxn_smi.split('>>')
        products = get_mol(p)

        assert len(bond_edits_graphs) == len(mol_list)
        if(len(mol_list)%args.batch_size==0) and len(mol_list):
            print(f'Saving after {total_examples}')
            sys.stdout.flush()
            batch_tensors = process_batch(bond_edits_graphs, mol_list, args)
            torch.save(batch_tensors, os.path.join(save_dir, f'batch-{num_batches}.pt'))

            num_batches = num_batches + 1
            mol_list = []
            bond_edits_graphs = []

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

        if (fragments is None) or (fragments.GetNumAtoms() <= 1):
            print(f"Fragments are invalid. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
            print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        tmp_frag = MultiElement(mol=Chem.Mol(fragments)).mols
        fragments = Chem.Mol()
        for mol in tmp_frag:
            fragments = Chem.CombineMols(fragments, mol)

        if len(reaction_info.core_edits) != 1:
            print('impossible')
            continue

        edit = reaction_info.core_edits[0]
        a1, a2, b1, b2 = edit.split(":")
        if float(b1) and float(b2) >= 0:
            bond_edits_graph = BondEditsRxn(prod_mol=Chem.Mol(products),
                                            frag_mol=Chem.Mol(fragments),
                                            reac_mol=Chem.Mol(reactants),
                                            edits_to_apply=[edit],
                                            rxn_class=reaction_info.rxn_class)

            frag_graph = MultiElement(mol=Chem.Mol(fragments))

            frag_mols = copy.deepcopy(frag_graph.mols)
            reac_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(reactants)).mols)
            bond_edits_graphs.append(bond_edits_graph)
            mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))
            total_examples += 1

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(info_all)} {mode} reactions processed.")
            sys.stdout.flush()

    print(f"All {mode} reactions complete.")
    sys.stdout.flush()

    assert len(bond_edits_graphs) == len(mol_list)
    batch_tensors = process_batch(bond_edits_graphs, mol_list, args)
    torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

    num_batches += 1
    mol_list = []
    bond_edits_graphs = []

    return num_batches



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--print_every', default=1000, type = int)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    num_batches = parse_bond_edits_forward(args=args, mode=args.mode)


if __name__ == '__main__':
    main()