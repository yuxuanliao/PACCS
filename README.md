# PACCS

This is the code repo for the paper *Accurate and Rational Collision Cross Section Prediction using Voxel Projected Area and Deep Learning*. We developed a Projected Area-based CCS prediction method (PACCS) directly from molecular conformers. PACCS supports users to generate large-scale and searchable CCS databases using the open-source Jupyter Notebook.

### Package required:
We recommend to use [conda](https://conda.io/docs/user-guide/install/download.html) and [pip](https://pypi.org/project/pip/).
- [python3](https://www.python.org/)
- [pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [RDKit](https://rdkit.org/)
- [PyTorch](https://pytorch.org/)
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/)

By using the [`requirements/conda/environment.yml`](requirements/conda/environment.yml) or [`requirements/pip/requirements.txt`](requirements/pip/requirements.txt) file, it will install all the required packages.

    git clone https://github.com/yuxuanliao/PACCS.git
    cd PACCS
    conda env create -f requirements/conda/environment.yml
    conda activate PACCS

## Data pre-processing
PACCS is a model for predicting CCS based on voxel projected area (vpa), so we need to convert SMILES strings to vpa. The related method is shown in [MolecularRepresentations.py](PACCS/MolecularRepresentations.py), [VoxelProjectedArea.py](PACCS/VoxelProjectedArea.py) and [MZ.py](PACCS/MZ.py)

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

**2.** Generate VPA. For details, see [VoxelProjectedArea.py](PACCS/VoxelProjectedArea.py). 

<img src="Voxel projected area.png" width:100px>

- Using the Fibonacci grids approach to distribute points evenly over the surfaces of 3D atomic spheres. 
- Projected on three coordinate planes (xy, xz, yz).
- Averaging.

## Model training
Train the model based on your own training dataset with [Training.py](PACCS/Training.py) function.

    PACCS_train(input_path, epochs, batchsize, output_model_path)

*Optionnal args*
- input_path : File path for storing the data of smiles and adduct.
- Parameters : Selected hyperparameters (epochs, batchsize).
- output_model_path : File path where the model is stored.

## Predicting CCS
The CCS prediction of the molecule is obtained by feeding the Graph and Adduct into the already trained SigmaCCS model with [Prediction.py](PACCS/Prediction.py).

    PACCS_predict(input_path, model_path, output_path)

*Optionnal args*
- input_path : File path for storing the data of smiles and adduct
- model_path : File path where the model is stored
- output_path : Path to save ccs prediction values

## Usage
The example code for model training is included in the [Model training.ipynb](Model%20training.ipynb). By directly running [train.ipynb](PACCS/train.ipynb), user can train the model based on your own training dataset.

The example code for CCS prediction is included in the [CCS prediction.ipynb](CCS%20prediction.ipynb). By directly running [Prediction.py](PACCS/Prediction.py), user can use PACCS to predict CCS values.

The example code for CCS prediction can use PACCS to predict CCS values by colab link [prediction.ipynb](https://colab.research.google.com/drive/1iln8N-JnBtywVOcLsHyKY-ImmpuMTXTc)).

## Information of maintainers
- 232303012@csu.edu.cn
- 1509200106@csu.edu.cn
