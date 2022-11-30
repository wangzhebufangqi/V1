import rdkit.Chem as Chem
import rdkit

rxn = '[N:1][c:2]1[c:3]([Br:4])[c:5][c:6]([Br:7])[c:8][c:9]1[C:10](=[O:11])[O:12]'
# [1,0,1.0,0.0; 1,0,1.0,0.0]
rxn = '[N-1:1][c:2]1[c:3]([Br:4])[c:5][c:6]([Br:7])[c:8][c:9]1[C:10](=[O:11])[O:12]'
# [1,0,1.0,0.0]
mol = Chem.MolFromSmiles(rxn)
charge_list = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
print(charge_list)