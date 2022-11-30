import torch
from rdkit import Chem
import argparse
import joblib
import os
import sys
import copy
from typing import Any
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.global_config import global_info_share_num
from seq_graph_retro.utils.parse import extract_leaving_groups
from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol
from seq_graph_retro.molgraph import BondEditsRxn, RxnElement, MultiElement
from seq_graph_retro.molgraph.vocab import Vocab
from seq_graph_retro.data.collate_fns import pack_graph_feats, prepare_lg_labels
from seq_graph_retro.utils import str2bool

DATA_DIR = '../datasets'

def process_batch_seq(edit_graphs, frag_batch, mol_list, args):
    assert len(edit_graphs) == len(frag_batch) == len(mol_list)
    lengths = torch.tensor([len(graph_seq) for graph_seq in edit_graphs], dtype=torch.long)
    max_seq_len = max([len(graph_seq) for graph_seq in edit_graphs])

    seq_tensors = []
    seq_labels = []
    mol_attrs = ['prod_mol']
    label_attrs = ['edit_label', 'h_label', 'done_label']
    seq_mask = []

    Main_ids = []
    Orders = []

    for idx in range(max_seq_len):
        graphs_idx = [copy.deepcopy(edit_graphs[i][min(idx, length-1)]).get_attributes(mol_attrs=mol_attrs,
                                                                label_attrs=label_attrs)
                     for i, length in enumerate(lengths)]
        mask = (idx < lengths).long()
        prod_graphs, edit_labels = list(zip(*graphs_idx))
        assert all([isinstance(graph, RxnElement) for graph in prod_graphs])

        orders = [copy.deepcopy(edit_graphs[i][min(idx,length-1)].get_order()) for i,length in enumerate(lengths)]

        if len(edit_labels[0]) == 1:
            edit_labels = torch.tensor(edit_labels, dtype=torch.long)
        else:
            edit_labels = [torch.tensor(edit_labels[i], dtype=torch.float) for i in range(len(edit_labels))]

        prod_tensors = pack_graph_feats(prod_graphs, directed=True, use_rxn_class=False)

        Orders.append(orders)
        seq_tensors.append(prod_tensors)
        seq_labels.append(edit_labels)
        seq_mask.append(mask)

    frag_tensors = pack_graph_feats(frag_batch, directed=True, use_rxn_class=False)
    seq_mask = torch.stack(seq_mask).long()
    assert seq_mask.shape[0] == max_seq_len
    assert seq_mask.shape[1] == len(mol_list)
    seq_order = torch.Tensor(Orders).long()
    assert seq_order.shape == seq_mask.shape

    return seq_tensors,seq_labels,seq_mask,frag_tensors,seq_order



def parse_bond_edits_seq(args):
    base_file = os.path.join(args.data_dir,f'{args.mode}','info')
    info_shards = global_info_share_num
    info_all = []
    for shard_num in range(info_shards):
        shard_file = base_file + f'-shard-{shard_num}'
        info_all.extend(joblib.load(shard_file))

    bond_edits_graphs = []
    bond_edits_frags = []
    mol_list = []
    save_dir = os.path.join(args.data_dir,f'{args.mode}','bond_edits_seq')

    os.makedirs(save_dir, exist_ok=True)

    num_batches = 0
    legal_cnt = 0

    for idx,reaction_info in enumerate(info_all):
        graph_seq = []
        rxn_smi = reaction_info.rxn_smi
        r, p = rxn_smi.split('>>')
        products = get_mol(p)

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

        if len(Chem.rdmolops.GetMolFrags(fragments)) != len(Chem.rdmolops.GetMolFrags(reactants)):
            print(f"Number of fragments don't match reactants. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        tmp_frag = MultiElement(mol=Chem.Mol(fragments)).mols
        fragments = Chem.Mol()
        for mol in tmp_frag:
            fragments = Chem.CombineMols(fragments, mol)

        edits_applied = []
        for _, edit in enumerate(reaction_info.core_edits):
            interim_mol = apply_edits_to_mol(Chem.Mol(products), edits_applied)
            if interim_mol is None:
                print("Interim mol is None")
                break
            graph = BondEditsRxn(prod_mol=Chem.Mol(interim_mol),
                                 frag_mol=Chem.Mol(fragments),
                                 reac_mol=Chem.Mol(reactants),
                                 edits_to_apply=[edit],
                                 rxn_class=reaction_info.rxn_class,
                                 order=reaction_info.Order
                                 )
            edits_applied.append(edit)
            graph_seq.append(graph)

        interim_mol = apply_edits_to_mol(Chem.Mol(products), edits_applied)
        # 实际上为 reactants  edits_to_apply=[]时done_label=1
        if interim_mol is not None:
            graph = BondEditsRxn(prod_mol=Chem.Mol(interim_mol),
                                 frag_mol=Chem.Mol(fragments),
                                 reac_mol=Chem.Mol(reactants),
                                 edits_to_apply=[],
                                 rxn_class=reaction_info.rxn_class,
                                 order=reaction_info.Order)

            frag_graph = MultiElement(mol=Chem.Mol(fragments),
                                      rxn_class=reaction_info.rxn_class)

            frag_mols = copy.deepcopy(frag_graph.mols)
            reac_mols = copy.deepcopy(MultiElement(mol=Chem.Mol(reactants)).mols)

            graph_seq.append(graph)
        else:
            continue

        if len(graph_seq) == 0:
            print(f"No valid fragment states found. Skipping reaction {idx}")
            print()
            sys.stdout.flush()
            continue

        bond_edits_graphs.append(graph_seq)
        bond_edits_frags.append(frag_graph)
        mol_list.append((products, copy.deepcopy(reac_mols), copy.deepcopy(frag_mols)))

        if (idx % args.print_every == 0) and idx:
            print(f"{idx}/{len(info_all)} {args.mode} reactions processed.")
            sys.stdout.flush()

        assert len(bond_edits_graphs) == len(bond_edits_frags) == len(mol_list)

        legal_cnt = legal_cnt + 1

        if (len(mol_list) % args.batch_size == 0) and len(mol_list):
            batch_tensors = process_batch_seq(bond_edits_graphs, bond_edits_frags, mol_list, args)
            torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))

            num_batches += 1
            bond_edits_frags = []
            bond_edits_graphs = []
            mol_list = []


    print(f"All {args.mode} reactions complete.")
    sys.stdout.flush()
    try:
        batch_tensors = process_batch_seq(bond_edits_graphs, bond_edits_frags, mol_list, args)
        print("Saving..")
        torch.save(batch_tensors, os.path.join(save_dir, f"batch-{num_batches}.pt"))
    except:
        print("last batch is empty")
    num_batches += 1
    bond_edits_frags = []
    bond_edits_graphs = []
    mol_list = []

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',default=DATA_DIR)
    parser.add_argument('--print_every', default=1000,type=int)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--batch_size',default=32,type=int)
    args = parser.parse_args()

    parse_bond_edits_seq(args)



if __name__=='__main__':
    main()
