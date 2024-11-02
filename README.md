# PACCS

This is the code repo for the paper *Accurate and Rational Collision Cross Section Prediction using Voxel Projected Area and Deep Learning*.  We developed a method named Projected Area-based CCS prediction method (PACCS), and a [dataset]…… including 8196 CCS values for three different ion adducts ([M+H]+, [M+Na]+ and [M-H]-). For each molecule, there are "SMILES", "Adduct", "ECCS", "grid_PA", "MW" and predicted CCS values of three adduct ion types. 

### Package required: 
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [rdkit](https://rdkit.org/)
- [PyTorch](https://pytorch.org/)

By using the [`requirements/conda/environment.yml`](requirements/conda/environment.yml), [`requirements/pip/requirements.txt`](requirements/pip/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/yuxuanliao/PACCS.git
    cd PACCS
    conda env create -f requirements/conda/environment.yml
    conda activate PACCS

## Data pre-processing
PACCS is a model for predicting CCS based on voxel projection area (VPA), so we need to convert SMILES strings to VPA. The related method is shown in ………….          

**1.** Generate 3D conformations of molecules. 

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = -1
    ps.maxAttempts = 1
    ps.numThreads = 0
    ps.useRandomCoords = True
    re = AllChem.EmbedMultipleConfs(mol, numConfs = 1, params = ps)
    re = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads = 0)
- [ETKDGv3](https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html?highlight=etkdgv3#rdkit.Chem.rdDistGeom.ETKDGv3) Returns an EmbedParameters object for the ETKDG method - version 3 (macrocycles).
- [EmbedMultipleConfs](https://www.rdkit.org/docs/source/rdkit.Chem.rdDistGeom.html?highlight=embedmultipleconfs#rdkit.Chem.rdDistGeom.EmbedMultipleConfs), use distance geometry to obtain multiple sets of coordinates for a molecule.
- [MMFFOptimizeMoleculeConfs](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html?highlight=mmffoptimizemoleculeconfs#rdkit.Chem.rdForceFieldHelpers.MMFFOptimizeMoleculeConfs), uses MMFF to optimize all of a molecule’s conformations

**2.** Generate VPA. For details, see …….    
- Using function fibonacci_sphere to get spheroidal coordinates of atoms.
- Projected on three coordinate planes (xy, xz, yz).
- Averaging.

## Model training
Train the model based on your own training dataset with …… function.

    Model_train(ifile, ParameterPath, ofile, ofileDataPath, EPOCHS, BATCHS, Vis, All_Atoms=[], adduct_SET=[])

*Optionnal args*
- ifile : File path for storing the data of smiles and adduct.
- ofile : File path where the model is stored.
- ParameterPath : Save path of related data parameters.
- ofileDataPath : File path for storing model parameter data.

## Predicting CCS
The CCS prediction of the molecule is obtained by feeding the Graph and Adduct into the already trained SigmaCCS model with …… function.

    Model_prediction(ifile, ParameterPath, mfileh5, ofile, Isevaluate = 0)

*Optionnal args*
- ifile : File path for storing the data of smiles and adduct
- ParameterPath : File path for storing model parameter data
- mfileh5 : File path where the model is stored
- ofile : Path to save ccs prediction values

## Usage
The example codes for usage is included in the [test.ipynb](test.ipynb)

## Information of maintainers
- zmzhang@csu.edu.cn
- 232303012@csu.edu.cn
- 1509200106@csu.edu.cn
