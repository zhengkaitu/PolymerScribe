import os
import requests
import time


from canon_tools import canonicalize_bigsmiles

validation_set = [
    ["test_network", "{[][$]CC=CC[$],[$]CC([<])C([<])C[$],[>]{[$][$]SS[$][$]}[>][]}"],
    ["test_dendrimer", "{[][>]C(=O)CCN(CCN[<])CCC(=O)[>][]}"],
    ["test_star", "C([#Arm])([#Arm])([#Arm])[#Arm].{#Arm=CO{[<][>]CCO[<][>]}}"],
    ["test_network2", "{[][<]C(=O)CCC(=O)[<],[>]OCCC(O[>])CO[>][]}"],
    ["test_network3", "{[][<]C(=O)CC(=O)[<],[>]OCC(CO[>])CO[>][]}"],
    ["test_star2", "OCCCCCC(=O){[>][<]OCCCCCC(=C)[>][<]}OCc1cc([#Arm1])cc([#Arm2])c1.{#Arm1=c2nnn(CC{[$][$]CC(c3ccccc3)[$][$]}CCC)c2}.{#Arm2=c4nnn({[>][<]CCO[>][<]}C)c4}"],
    ["test_dendrimer", "{[][>2]C(=O)CC(=O)[<1],[>3]OCC(CO[<2])CO[<2],[>2]C(=O)CC(=O)[<3],[>3]OCC(CO[<1])CO[<1][>1]}"],
    ["test_block", "{[][<]OO[>][>]}{[>][<]CC[>][]}"],
    ["test_dendrimer", "{[][>2]C(=O)CC(=O)[<2],[>3]OCC(CO[<2])CO[<2],[>2]C(=O)CC(=O)[<3],[>3]OCC(CO[<2])CO[<2][>2]}"],
    ["test_dendrimer2", "{[][<]C(=O)CC(=O)[<],[>]OCC(CO[>])CO[>][]}"],
    ["forward_dendrimer", "{[][>2]C(=O)CC(=O)[<1],[>3]OCC(CO[<2])CO[<2],[>2]C(=O)CC(=O)[<3],[>3]OCC(CO[<1])CO[<1][>1]}"],
    ["reverse_dendrimer",
     "{[>1][>2]C(=O)CC(=O)[<1],[>3]OCC(CO[<2])CO[<2],[>2]C(=O)CC(=O)[<3],[>3]OCC(CO[<1])CO[<1][]}"],
    ["test_star", "C(C{[<][>]OCC[<][>]}O)(CO{[<][<]OCC[>][>]})(C{[<][>]OCC[<][>]}O)CO{[<][<]OCC[>][>]}"],
    ["test_star2", "C(CO{[<][>]CCO[<][>]})(CO{[<][>]CCO[<][>]})(CO{[<][>]CCO[<][>]})CO{[<][>]CCO[<][>]}"],
    ["many_endgroups_real", "N#CC(C)(C){[$][$]CC(c1ccccc1)[$];[$]C(C)(C)C#N,[$]C=C(c1ccccc1)[]}"],
    ["many_endgroups_real2", "[H]CC(C#N)(C){[<][>]CC(c(cc1)ccc1)[<],[>]C(c(cc1)ccc1)C[<];[>]C(C#N)(C)C[H],[>]C=Cc(cc1)c(cc1)[H][]}"],
    ["many_endgroups", "{[][<]CC[>];[<]OO,[<]NN,[>]SS[]}"],
    ["many_endgroups_2", "{[>][<]CC[>];[<]OO,[<]{[>][<]SS[>][<]}NN[]}"],
    ["implicit_endgroup_star", "{[>][<]CC[>];[<]C({[>][<]SS[>][<]}NN){[>][<]OO[>][<]}[]}"],
    ["multiple_implicit_endgroup_star", "{[][<]CC[>];[<]C({[>][<]SS[>][<]}){[>][<]OO[>][<]},[>]O,[>]N[]}"],
    ["multiple_arm_implicit_endgroup_star", "{[>][<]CC[>];[<]C({[>][<]SS[>][<]})C({[>][<]OO[>][<]}){[>][<]SS[>][<]}[]}"],
    ["implicit_endgroups", "{[>][<]CCO[>];[<]CCO{[$][$]CC(c1ccccc1)[$][]}[]}"],
    ["test_network", "Br{[<][<]OC(S[<])(O[<])O[<],[>]NN[>][>]}N"],
    ["test_block", "{[][<]C(C(=O)O)C[>],[<]CC(C(=O)O)[>],[<]CC(C(=O)O{[>][<]CCO[>][<]}C)[>],[<]C(C(=O)O{[>][<]CCO[>][<]}C)C[>][]}"],
    ["test_block2",
     "{[][$]CC(C(=O)O)[$],[$]CC(C(=O)O{[>]C(C[<])O[>][<]}C)[$][]}"],
    ["block1", "CCO{[>][<]CCO[>][<]}{[$][$]CC(c1ccccc1)[$][]}"],
    ["block2", "{[>][<]CCO[>],[<]CCO[>2],[<2]CC(c1ccccc1)[>2],[>2]CC(c1ccccc1)[<2][<2]}"],
    ["block3", "C{[>][<]COC[>][<]}CO{[$][$]CC(c1ccccc1)[$][]}"],
    ["PEG", "CCO{[>][<]CCO[>][<]}CCO"],
    ["Single object with end groups", "C=CC(=O){[<][>]OCC[<][>]}OC(=O)C=C"],
    ["Single object without end groups", "{[][<]OCCO[<],[>]C(=O)CCC(=O)[>][]}"],
    ["Di-block polymer", "CCC(C){[$][$]CC(C1CCCCC1)[$][$]}{[$][$]CCCC[$],[$]CC(CC)[$][$]}[H]"],
    ["Tri-block polymer", "CCC(C){[$][$]CC(c1ccccc1)[$][$]}{[$][$]CC=C(C)C[$],[$]CC(C(C)=C)[$],[$]CC(C)(C=C)[$][$]}{[$][$]CC(c1ccccc1)[$][$]}{[>][<]CCO[>][<]}[H]"],
    ["Segmented polymer", "{[][<]N=Cc(cc1)ccc1C=NCCC[Si](C)O{[<][>][Si](C)O[<][>]}[Si](C)CCC[>][]}"],
    ["Graft polymer", "COCCO{[>][<]C(O{[<][>]CCO[<][>]}C)CCCCCO[>],[<]C(=O)CCCCCO[>][<]}"],
    ["6-armed dendrimer", "N(CCN([#R])([#R]))(CCN([#R])([#R]))(CCN([#R])([#R])).{#R=CCC(=O){[>][<]NCCN(CCC(=O)[>])CCC(=O)[>][]}}"],
    ["Dendrimer", "{[][<]C(=O)CC(=O)[<],[>]OCC(CO[>])CO[>][]}"],
    ["4-armed star polymer", "C([#Arm])([#Arm])([#Arm])[#Arm].{#Arm=CO{[<][>]CCO[<][>]}}"],
    ["3-armed star polymer", "OCCCCCC(=O){[>][<]OCCCCCC(=C)[>][<]}OCc1cc([#Arm1])cc([#Arm2])c1.{#Arm1=c2nnn(CC{[$][$]CC(c3ccccc3)[$][$]}CCC)c2}.{#Arm2=c4nnn({[>][<]CCO[>][<]}C)c4}"],
    ["Vulcanized polymer", "{[][$]CC=CC[$],[$]CC([<])C([<])C[$],[>]{[$][$]SS[$][$]}[>][]}"],
    ["Polymer network", "{[][>]C(=O)CCCCCCC(=O)[>],C([#R])([#R])OC([#R])([#R])[]}.{#R=COC(CO{[<][>]CCO[<][>]}CCN[<])(CO{[<][>]CCO[<][>]}CCN[<])}"],
    ["PEG", "CCO{[>][<]CCO[>][<]}CCO"],
    ["macrocycle1", "C1CO{[>][<]CCO[>][<]}CCO1"],
    ["macrocycle2", "O1CC{[>][<]OCC[>][<]}OCC1"],
    ["test", "{[][>0]CC(c(cc1)ccc1)[<0],[>0]C(c(cc1)ccc1)C[<0];[H]{[<][>]CC(C)=CC[<][>]}[<0][]}"],
    ["block4", "{[>][<]CCO[>][<]}CCO{[$][$]CC(c1ccccc1)[$][]}"],
    ["block5", "{[>][<]CCO[>][<]}CCO{[>][<]CCO[>][<]}{[$][$]CC(c1ccccc1)[$][]}"],
    ["polystyrene", "c1ccccc1CC{[>][<]CC(c1ccccc1)[>][<]}"],
    ["frameshifted_star1", "O{[<][>]CCO[<][>]}CC(C{[<][>]OCC[<][>]}O)(C{[<][>]OCC[<][>]}O)C{[<][>]OCC[<][>]}O"],
    ["frameshifted_star2", "O{[<][>]CCO[<][>]}CC({[<][>]COC[<][>]}CO)(C{[<][>]OCC[<][>]}O)C{[<][>]OCC[<][>]}O"],
    ["frameshifted_star3", "C([#Arm])([#Arm])([#Arm])[#Arm].{#Arm=CO{[<][>]CCO[<],[>]CCO[<][>]}}"],
    ["tacticity", "{[][<]O/C=C/O[>][]}"],
    ["test_star", "{[][<]OO[>][<]}C({[>][<]OO[>][]}){[>][<]OO[>][]}"],
    ["implicit_ends", "{[][<]CCO[>];[>]OCC[<]}CCO"],
    ["implicit_ends_2", "{[][<]CCO[>];[>]OCC,[>]C#N[<]}CCO"],
    ["implicit_ends_3", "Br{[<][<]OC(S[<])(O[<])O[<],[>]NN[>][>]}N"],
    ["implicit_ends_4", "{[][<]OC(S[<])(O[<])O[<],[>]NN[>];[>]N,[<]Br[]}"],
    ["implicit_ends_5", "{[][<]OC(S[<])(O[<])O[<],[>]NN[>];[>]N,[<]Br,O[>][]}"],
    ["test_graft", "{[][>]CC(O{[>][<]OO[>][]})[<][]}"],
    ["test_split", "CCO{[>][<]C[>2],[<2]CO[>][<]}CCO"],
    ["initial", "N#CC(C)(C){[$][$]CC(c1ccccc1)[$];[$]C(C)(C)C#N,[$]C=C(c1ccccc1)[]}"],
    ["second_canonicalization2", "[H]CC(C#N)(C){[<][>]CC(c(cc1)ccc1)[<],[>]C(c(cc1)ccc1)C[<];[>]C(C#N)(C)C[H],[>]C=Cc(cc1)c(cc1)[H][]}"],
]


def test():
    for subfolder_name, bigsmiles in validation_set:
        start = time.time()

        print(f"Canonicalizing {bigsmiles}", flush=True)

        # canonical = canonicalize_bigsmiles(
        #     bigsmiles=bigsmiles,
        #     output_folder=os.path.join("Output", subfolder_name),
        #     plot=False
        # )
        #
        # print(f"Time: {time.time() - start: .2f} s    {canonical}", flush=True)

        try:
            resp = requests.post(
                url="http://0.0.0.0:3319/canonicalize-bigsmiles/",
                json={"bigsmiles": bigsmiles},
                timeout=300
            )

            print(f"Time: {time.time() - start: .2f} s    {resp.json()}", flush=True)
        except requests.exceptions.Timeout:
            print("The request timed out!")


if __name__ == "__main__":
    test()
