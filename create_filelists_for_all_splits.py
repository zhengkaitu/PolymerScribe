import glob
import os
import random
from rdkit import Chem


def verify_labels(sources: list[str]):
    for source in sources:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        basenames = [fn.split("/")[-1].rstrip(".png") for fn in fl]

        assert all(os.path.exists(f"{source}{basename}.corrected.mol") for basename in basenames)
        print(f"All corrected for {source}. Number of images: {len(basenames)}")


def get_filelist_fully_random(sources_generic, sources_ladder):
    filelist_generic_all = []
    for source in sources_generic:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        filelist_generic_all.extend(fl)
    random.shuffle(filelist_generic_all)

    filelist_ladder_all = []
    for source in sources_ladder:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        filelist_ladder_all.extend(fl)
    random.shuffle(filelist_ladder_all)

    filelist_generic_train_realistic = filelist_generic_all[:760]
    filelist_generic_val = filelist_generic_all[760:855]
    filelist_generic_test = filelist_generic_all[855:]

    filelist_ladder_train_realistic = filelist_ladder_all[:40]
    filelist_ladder_val = filelist_ladder_all[40:45]
    filelist_ladder_test = filelist_ladder_all[45:]

    assert len(filelist_generic_train_realistic) == 760
    assert len(filelist_generic_val) == 95
    assert len(filelist_generic_val) == 95
    assert len(filelist_ladder_train_realistic) == 40
    assert len(filelist_ladder_val) == 5
    assert len(filelist_ladder_test) == 5

    filelist_train_realistic = filelist_generic_train_realistic + filelist_ladder_train_realistic
    filelist_val = filelist_generic_val + filelist_ladder_val
    filelist_test = filelist_generic_test + filelist_ladder_test

    random.shuffle(filelist_train_realistic)

    return filelist_train_realistic, filelist_val, filelist_test


def get_filelist_docwise_random(sources_generic, sources_ladder):
    filelist_generic_all = {}
    for source in sources_generic:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        for fn in fl:
            doi = fn.split("/")[-1].split("_")[0]
            key = (source, doi)
            if key in filelist_generic_all:
                filelist_generic_all[key].append(fn)
            else:
                filelist_generic_all[key] = [fn]

    items = list(filelist_generic_all.items())
    random.shuffle(items)

    filelist_generic_train_realistic = []
    filelist_generic_val = []
    filelist_generic_test =[]
    for (source, doi), fns in items:
        if len(filelist_generic_val) + len(fns) <= 95:
            filelist_generic_val.extend(fns)
            continue

        if len(filelist_generic_test) + len(fns) <= 95:
            filelist_generic_test.extend(fns)
            continue

        filelist_generic_train_realistic.extend(fns)

    filelist_ladder_all = {}
    for source in sources_ladder:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        for fn in fl:
            doi = fn.split("/")[-1].split("_")[0]
            key = (source, doi)
            if key in filelist_ladder_all:
                filelist_ladder_all[key].append(fn)
            else:
                filelist_ladder_all[key] = [fn]

    items = list(filelist_ladder_all.items())
    random.shuffle(items)

    filelist_ladder_train_realistic = []
    filelist_ladder_val = []
    filelist_ladder_test =[]
    for (source, doi), fns in items:
        if len(filelist_ladder_val) + len(fns) <= 5:
            filelist_ladder_val.extend(fns)
            continue

        if len(filelist_ladder_test) + len(fns) <= 5:
            filelist_ladder_test.extend(fns)
            continue

        filelist_ladder_train_realistic.extend(fns)

    assert len(filelist_generic_train_realistic) == 760
    assert len(filelist_generic_val) == 95
    assert len(filelist_generic_val) == 95
    assert len(filelist_ladder_train_realistic) == 40
    assert len(filelist_ladder_val) == 5
    assert len(filelist_ladder_test) == 5

    filelist_train_realistic = filelist_generic_train_realistic + filelist_ladder_train_realistic
    filelist_val = filelist_generic_val + filelist_ladder_val
    filelist_test = filelist_generic_test + filelist_ladder_test

    random.shuffle(filelist_train_realistic)

    return filelist_train_realistic, filelist_val, filelist_test


def get_filelist_docwise_atom_count(sources_generic, sources_ladder):
    def _get_average_atom_count(_tup):
        _key, _fns = _tup
        _atom_counts = []
        for _fn in _fns:
            mol_fn = _fn.replace(".png", ".corrected.mol")
            mol = Chem.MolFromMolFile(mol_fn, sanitize=False, removeHs=False)
            _atom_counts.append(mol.GetNumHeavyAtoms())

        return sum(_atom_counts) / len(_atom_counts)

    filelist_generic_all = {}
    for source in sources_generic:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        for fn in fl:
            doi = fn.split("/")[-1].split("_")[0]
            key = (source, doi)
            if key in filelist_generic_all:
                filelist_generic_all[key].append(fn)
            else:
                filelist_generic_all[key] = [fn]

    items = list(filelist_generic_all.items())
    items = sorted(items, key=_get_average_atom_count, reverse=True)

    filelist_generic_train_realistic = []
    filelist_generic_val = []
    filelist_generic_test =[]
    for (source, doi), fns in items:
        if len(filelist_generic_test) + len(fns) <= 95:
            print(f"Generic, test, {source}, {doi}, average atom count: {_get_average_atom_count((None, fns))}")
            filelist_generic_test.extend(fns)
            continue

        if len(filelist_generic_val) + len(fns) <= 95:
            print(f"Generic, val, {source}, {doi}, average atom count: {_get_average_atom_count((None, fns))}")
            filelist_generic_val.extend(fns)
            continue

        print(f"Generic, train, {source}, {doi}, average atom count: {_get_average_atom_count((None, fns))}")
        filelist_generic_train_realistic.extend(fns)

    filelist_ladder_all = {}
    for source in sources_ladder:
        fl = sorted(glob.glob(os.path.join(source, "*.png")))
        for fn in fl:
            doi = fn.split("/")[-1].split("_")[0]
            key = (source, doi)
            if key in filelist_ladder_all:
                filelist_ladder_all[key].append(fn)
            else:
                filelist_ladder_all[key] = [fn]

    items = list(filelist_ladder_all.items())
    items = sorted(items, key=_get_average_atom_count, reverse=True)

    filelist_ladder_train_realistic = []
    filelist_ladder_val = []
    filelist_ladder_test =[]
    for (source, doi), fns in items:
        if len(filelist_ladder_test) + len(fns) <= 5:
            print(f"Ladder, test, {source}, {doi}, average atom count: {_get_average_atom_count((None, fns))}")
            filelist_ladder_test.extend(fns)
            continue

        if len(filelist_ladder_val) + len(fns) <= 5:
            print(f"Ladder, val, {source}, {doi}, average atom count: {_get_average_atom_count((None, fns))}")
            filelist_ladder_val.extend(fns)
            continue

        print(f"Ladder, train, {source}, {doi}, average atom count: {_get_average_atom_count((None, fns))}")
        filelist_ladder_train_realistic.extend(fns)

    assert len(filelist_generic_train_realistic) == 760
    assert len(filelist_generic_val) == 95
    assert len(filelist_generic_val) == 95
    assert len(filelist_ladder_train_realistic) == 40
    assert len(filelist_ladder_val) == 5
    assert len(filelist_ladder_test) == 5

    filelist_train_realistic = filelist_generic_train_realistic + filelist_ladder_train_realistic
    filelist_val = filelist_generic_val + filelist_ladder_val
    filelist_test = filelist_generic_test + filelist_ladder_test

    random.shuffle(filelist_train_realistic)

    return filelist_train_realistic, filelist_val, filelist_test


def create_splits(split: str, sources_synthetic, sources_generic, sources_ladder):
    output_path = f"experiments/{split}"
    os.makedirs(output_path, exist_ok=True)

    if split == "fully_random":
        filelist_train_realistic, filelist_val, filelist_test = \
            get_filelist_fully_random(sources_generic, sources_ladder)
    elif split == "docwise_random":
        filelist_train_realistic, filelist_val, filelist_test = \
            get_filelist_docwise_random(sources_generic, sources_ladder)
    elif split == "docwise_atom_count":
        filelist_train_realistic, filelist_val, filelist_test = \
            get_filelist_docwise_atom_count(sources_generic, sources_ladder)
    else:
        raise NotImplementedError(f"Split {split} not implemented")

    filelist_train_synthetic = sources_synthetic
    for realistic_count in [0, 200, 400, 600, 800]:
        ofn = os.path.join(output_path, f"{realistic_count}_train.filelist.txt")
        with open(ofn, "w") as of:
            for fl in filelist_train_synthetic:
                of.write(f"{fl}\n")

            for fn in sorted(filelist_train_realistic[:realistic_count]):
                of.write(f"{fn}\n")

    ofn = os.path.join(output_path, "val.filelist.txt")
    with open(ofn, "w") as of:
        for fn in sorted(filelist_val):
            of.write(f"{fn}\n")

    ofn = os.path.join(output_path, "test.filelist.txt")
    with open(ofn, "w") as of:
        for fn in sorted(filelist_test):
            of.write(f"{fn}\n")


def main():
    random.seed(0)

    sources_synthetic = [
        "data/mt_images_processed/",
        "data/olsen_images_processed/bigsmiles_manuscript/",
        "data/olsen_images_processed/bigsmiles_si/",
        "data/olsen_images_processed/canonicalization_manuscript/",
        "data/olsen_images_processed/canonicalization_si/",
        "data/olsen_images_processed/non-covalent_manuscript/",
        "data/olsen_images_processed/non-covalent_si/"
    ]

    sources_generic = [
        "data/realistic_images_processed/generic/acspolymersau/",
        "data/realistic_images_processed/generic/acsmacrolett/",
        "data/realistic_images_processed/generic/macromolecules/"
    ]

    sources_ladder = [
        "data/realistic_images_processed/ladder/acsmacrolett/",
        "data/realistic_images_processed/ladder/angewchemie/",
        "data/realistic_images_processed/ladder/chemengjournal/",
        "data/realistic_images_processed/ladder/chemicalscience/",
        "data/realistic_images_processed/ladder/digitaldiscovery/",
        "data/realistic_images_processed/ladder/faradaydiscussions/",
        "data/realistic_images_processed/ladder/macromolecules/",
        "data/realistic_images_processed/ladder/polymer/"
    ]

    verify_labels(sources=sources_synthetic)
    verify_labels(sources=sources_generic)
    verify_labels(sources=sources_ladder)

    create_splits("fully_random", sources_synthetic, sources_generic, sources_ladder)
    create_splits("docwise_random", sources_synthetic, sources_generic, sources_ladder)
    create_splits("docwise_atom_count", sources_synthetic, sources_generic, sources_ladder)


if __name__ == "__main__":
    main()
