# Common molecules for dataset diversity
COMMON_MOLECULES = [
    # Simple organic molecules
    ("CC(=O)O", "acetic_acid"),
    ("CCO", "ethanol"),
    ("CC(C)O", "isopropanol"),
    ("C1=CC=CC=C1", "benzene"),
    ("CC(=O)C", "acetone"),
    # Amino acids (with stereochemistry) - L forms
    ("N[C@@H](C)C(=O)O", "L-alanine"),
    ("N[C@@H](CC(=O)O)C(=O)O", "L-aspartic_acid"),
    ("N[C@@H](CCC(=O)O)C(=O)O", "L-glutamic_acid"),
    ("N[C@@H](CC1=CC=CC=C1)C(=O)O", "L-phenylalanine"),
    ("N[C@@H](CC(C)C)C(=O)O", "L-leucine"),
    ("N[C@@H](CO)C(=O)O", "L-serine"),
    ("N[C@@H](C(C)O)C(=O)O", "L-threonine"),
    ("N[C@@H](CS)C(=O)O", "L-cysteine"),
    ("N[C@@H](CCSC)C(=O)O", "L-methionine"),
    ("N[C@@H](CCCCN)C(=O)O", "L-lysine"),
    # Amino acids - D forms for variation
    ("N[C@H](C)C(=O)O", "D-alanine"),
    ("N[C@H](CC(=O)O)C(=O)O", "D-aspartic_acid"),
    ("N[C@H](CO)C(=O)O", "D-serine"),
    # Sugars (with stereochemistry)
    ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "D-glucose"),
    ("OC[C@H]1OC(CO)(O)[C@@H](O)[C@@H]1O", "D-fructose"),
    ("OC[C@H]1O[C@H](O)[C@H](O)[C@@H]1O", "D-ribose"),
    ("C[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "L-rhamnose"),
    ("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O", "D-galactose"),
    # Nucleotides
    ("C1=NC2=C(N1)C(=O)NC(=N2)N", "guanine"),
    ("CC1=CN(C(=O)NC1=O)C2CC(C(O2)CO)O", "thymidine"),
    ("C1=CN(C=O)C(=O)NC1=N", "cytosine"),
    # Pharmaceuticals with stereochemistry
    ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
    ("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "S-ibuprofen"),
    ("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O", "R-ibuprofen"),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
    ("C[C@H](Cc1ccccc1)NC", "S-methamphetamine"),
    ("CC(C)NCC(O)c1ccc(O)c(CO)c1", "salbutamol"),
    # Chiral alcohols and acids
    ("C[C@H](O)C(=O)O", "L-lactic_acid"),
    ("C[C@@H](O)C(=O)O", "D-lactic_acid"),
    ("C[C@H](O)CCO", "S-1,3-butanediol"),
    ("CC[C@H](C)O", "S-2-butanol"),
    ("CC[C@@H](C)O", "R-2-butanol"),
    ("C[C@H](O)c1ccccc1", "S-1-phenylethanol"),
    ("C[C@@H](O)c1ccccc1", "R-1-phenylethanol"),
    # Chiral ethers and cyclic compounds
    ("C[C@@H]1CCCO1", "R-2-methyltetrahydrofuran"),
    ("C[C@H]1CCCO1", "S-2-methyltetrahydrofuran"),
    ("C[C@H]1CCCCO1", "S-2-methyltetrahydropyran"),
    ("O[C@H]1CCCC[C@H]1O", "trans-1,2-cyclohexanediol"),
    # Terpenes and natural products with stereochemistry
    ("C[C@H]1CC[C@@H](CC1)C(C)=C", "R-limonene"),
    ("CC(C)=CCC[C@@H](C)C1CC=C(C)C(=O)C1", "S-carvone"),
    ("C[C@H]1CCC(=C(C)C)C[C@H]1O", "menthol"),
    # Note: bornyl acetate removed - bridged bicyclic causes rendering issues
    # Aromatic compounds
    ("c1ccc2c(c1)ccc3c2cccc3", "anthracene"),
    ("c1ccc2c(c1)ccc1c2cccc1", "phenanthrene"),
    ("CC1=CC=C(C=C1)C", "para-xylene"),
    # Alcohols and ethers
    ("CCCCO", "butanol"),
    ("CCOC(=O)C", "ethyl_acetate"),
    ("CCOCC", "diethyl_ether"),
    # Aldehydes and ketones
    ("CC=O", "acetaldehyde"),
    ("O=Cc1ccccc1", "benzaldehyde"),
    ("CC(=O)CC(C)C", "methyl_isobutyl_ketone"),
    # Amines
    ("CCN", "ethylamine"),
    ("c1ccc(cc1)N", "aniline"),
    ("CN(C)C", "trimethylamine"),
    # Carboxylic acids
    ("CCCC(=O)O", "butyric_acid"),
    ("c1ccc(cc1)C(=O)O", "benzoic_acid"),
    # Heterocycles
    ("c1cccnc1", "pyridine"),
    ("c1cnc[nH]1", "imidazole"),
    ("c1ccc2[nH]ccc2c1", "indole"),
    # Additional diversity
    ("CC(C)(C)O", "tert-butanol"),
    ("c1ccc(cc1)O", "phenol"),
    ("CC(C)C(=O)C(C)C", "diisobutyl_ketone"),
    ("CCCCCC", "hexane"),
    ("C1CCCCC1", "cyclohexane"),
    ("CC1=CC=C(C=C1)O", "para-cresol"),
]
