from typing import List
import re


ORGANIC_SET = {'B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}

RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']

# PLACEHOLDER_ATOMS = ["Xe", "Rn", "Nd", "Yb", "At", "Fm", "Er"]
PLACEHOLDER_ATOMS = ["Lv", "Lu", "Nd", "Yb", "At", "Fm", "Er"]


'''
Define common substitutions for chemical shorthand
Note: does not include R groups or halogens as X
'''
class Substitution(object):

    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability


SUBSTITUTIONS: List[Substitution] = [
    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),

    Substitution(['OAc'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.8),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.8),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),

    Substitution(['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  # Benzoyl
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  # Benzoyl

    Substitution(['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  # Benzyl
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  # Benzyl

    Substitution(['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.9),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.9),
    Substitution(['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),

    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                           "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.8),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.7),
    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),  # what's probability? what's SMARTS?
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['SEM'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),  # what's probability? what's SMARTS?
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.8),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS'], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  # Tos

    # Alkyl chains
    Substitution(['OMe', 'MeO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[NH]C", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),

    # Branched
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.7),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  # COOH
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    Substitution(['OCH3', 'H3CO'], '[OH0;D2][CH3]', "[O]C", 0.4),
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),
]


ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS for abbrv in sub.abbrvs}

VALENCES = {"H": [1],
"Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3,5], "O": [2], "F": [1],
"Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5,3], "S": [6,2,4], "Cl": [1],
"K": [1], "Ca": [2], "Sc": [3], "Ti": [4,3,2], "V": [5,4,3,2], "Cr": [6,3,2], "Mn": [7,6,4,3,2], "Fe": [3,2], "Co": [3,2], "Ni": [2], "Cu": [2,1], "Zn": [2], "Ga": [3], "Ge": [4,2], "As": [5,3], "Se": [6,4,2], "Br": [1], "Kr": [2,1],
"Rb": [1], "Sr": [2], "Y": [3], "Zr": [4], "Nb": [5], "Mo": [6,4], "Tc": [7,4], "Ru": [4,3], "Rh": [3], "Pd": [4,2,0], "Ag": [1], "Cd": [2], "In": [3], "Sn": [4,2], "Sb": [5,3], "Te": [6,4,2], "I": [1], "Xe": [8,6,4,2],
"Cs": [1], "Ba": [2], "La": [3], "Ce": [4,3], "Pr": [3], "Nd": [3], "Pm": [3], "Sm": [3], "Eu": [3,2], "Gd": [3], "Tb": [3], "Dy": [3], "Ho": [3], "Er": [3], "Tm": [3], "Yb": [3], "Lu": [3], "Hf": [4], "Ta": [5], "W": [6,4], "Re": [4], "Os": [4], "Ir": [4,3], "Pt": [4,2], "Au": [3,1], "Hg": [2,1], "Tl": [3,1], "Pb": [4,2], "Bi": [3], "Po": [4,2], "At": [1], "Rn": [2],
"Fr": [1], "Ra": [2], "Ac": [3], "Th": [4], "Pa": [5], "U": [6,4], "Np": [5], "Pu": [4], "Am": [3], "Cm": [3], "Bk": [3], "Cf": [3], "Es": [3], "Fm": [3], "Md": [3], "No": [2], "Lr": [3], "Rf": [4], "Db": [5], "Sg": [6], "Bh": [7], "Hs": [8], "Cn": [2],
"Ac": [1], "Bz": [1], "Bn": [1], "Boc": [1], "Cbm": [1], "Cbz": [1], "Cy": [1], "Fmoc": [1], "Mes": [1], "Ms": [1], "Ph": [1], "PMB": [1], "Py": [1], "SEM": [1], "Suc": [1], "TBS": [1], "TBZ": [1], "Tf": [1], "TFA": [1], "TMS": [1], "Ts": [1],
"Me": [1], "Et": [1], "Pr": [1], "nPr": [1], "n-Pr": [1], "iPr": [1], "i-Pr": [1], "Bu": [1], "nBu": [1], "n-Bu": [1], "iBu": [1], "i-Bu": [1], "tBu": [1], "t-Bu": [1]
}

# tokens of condensed formula
FORMULA_REGEX = re.compile('(' + '|'.join(VALENCES.keys()) + '|[0-9]+|\(|\))')
# abbreviations that don't follow the pattern [A-Z][a-z]*
SPECIAL_ABBREV = ["PMB", "SEM", "TBS", "TBZ", "TFA", "TMS", "n-?Pr", "i-?Pr", "n-?Bu", "i-?Bu", "t-?Bu", "R[0-9]*"]
# tokens of condensed formula (extended)
FORMULA_REGEX_BACKUP = re.compile('(' + '|'.join(SPECIAL_ABBREV) + "|[A-Z][a-z]*|[0-9]+|\(|\))")