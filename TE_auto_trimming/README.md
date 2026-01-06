# A Deep Learning–Based Computational Tool for Automatic Trimming of Transposable Elements in Large-Scale Genomic Projects
Based on a computer vision strategy, this project proposes the application of deep learning techniques, specifically convolutional neural networks, for the identification and trimming of transposable elements (TEs).
To create the input for the convolutional neural networks, FASTA sequences are converted into PDF files and then into images using the tool TE-Aid (Goubert et al., 2022).
Image data are stored in NumPy arrays (dataset) with the corresponding labels (for synthetic data), for model training and testing.
Also, the tool allows to input a FASTA file with sequences for trimming, returning the trimmed sequences as output. 

The tool consists of three Python scripts:
- `Auto_trimming.py`, which includes a basic framework for loading the data, getting the model and running experiments.
- `dataset_library`: contains the functions to generate PDF files and images from the sequences of the FASTA file.
- `model_library`: contains the functions for training and testing of models. 

## Installation
1. Clone the repository:
```
git clone https://github.com/[username]/TE_auto_trimming.git
```

2. Install TE-Aid from github
```
git clone https://github.com/clemgoub/TE-Aid.git TEAid
```
In order for TE-Aid to work, the parameter `-num_threads 16` needs to be added to blastp:
```
blastp -query $OUTPUT/TE.orfs -db $DIR/db/RepeatPeps.lib -outfmt 6 -num_threads 16 | sort -k1,1 -k12,12nr | sort -u -k1,1 | sed 's/#/--/g' > $OUTPUT/TE.blastp.out
```

3. Create the environment from the environment.yml file:
```
mamba env create -f autotrim_env.yml

mamba activate autotrim_env
```

## Pipeline workflow
This computational tool offers two options:
- Option 1: Training a model and testing with synthetic data, generated, for example, from a curated library of TE sequences.
- Option 2: TE trimming from an input FASTA file containing sequences to curate.

For both options, FASTA headers must follow a specific format, with the species indicated at the end of the header, separated by a space from the rest of the text. 
In this way, the species will be recognized and its genome will be downloaded, which is required to run TE-Aid. Example:
```
>DR000395818#CLASSI/LINE/CR1 Bucorvus abyssinicus
```

### Option 1: Generation of synthetic data
This option generates sequences containing one or two TEs, combined with randomly generated DNA sequences, for a total length of 15,000 bp. This script does not require a specific environment.

**Batch processing (HPC/SLURM):**
```batch
sbatch data_generation/run_generation.sh
```
**Executes** `data_generation/GenerationData.py`

**Output**
It will generate a FASTA file `simulated_data_merged.fasta`.

**Parameters**
- `--fasta`: Path to FASTA file from which synthetic data will be generated (required)
- `--seq_per_case`: number of sequences to generate per case (4 cases)

For next steps after synthetic data generation, the `autotrim_env` environment is required.

### Create images from FASTA file with TE-Aid
For this step, the following scripts from Goubert et al. (2022) are needed: TE-Aid, Run-c2g.R, consensus2genome.R and blastndotplot.R.
By default, they will be included in the folder `TEAid`.

**Batch processing (HPC/SLURM):**
```batch
sbatch run_teaid.sh
```
**Executes** `Auto_trimming.py with --mode teaid`

**Parameters**
- `--input_fasta`: Path to FASTA file from which the images will be created (required)

**Output**
It will create a folder called `te_aid`, in which the PDF files and images will be generated. 

### Create a dataset

**Batch processing (HPC/SLURM):**
```batch
sbatch create_dataset.sh
```
**Executes** `Auto_trimming.py with --mode dataset`

**Parameters**
- `--input_fasta`: Path to FASTA file from which the dataset will be created (required)
- `--dataset_dir`: Directory where the dataset will be saved

**Output**
In the specified directory, this will generate 4 Numpy matrices: features_data.npy, labels_data.npy, case_labels.npy and species_labels.npy.

After generating the PDFs (and images) and the dataset, we can choose in the script Auto_trimming.py if we want to do training or testing (for option 1) or trimming (for option 2). 

### Training

**Batch processing (HPC/SLURM):**
```batch
sbatch auto_trimming.sh
```
**Executes** `Auto_trimming.py with --mode train`

**Parameters**
- `--dataset_dir`: Directory where the dataset is saved

**Output**
- trained_model.h5
- X_test.npy and Y_test.npy
- scalerX.bin
- model.weights.h5 (inside `checkpoint` folder)
- Performance plots: Train_Curve_Loss.png (Epoch vs MSE) and Train_Curve_R2.png (Epoch vs R2)

### Testing

**Batch processing (HPC/SLURM):**
```batch
sbatch auto_trimming.sh
```
**Executes** `Auto_trimming.py with --mode test`

**Parameters**
- `--model`:
- `--scaler`:
- `--dataset_testing`: Directory where the dataset for testing is saved

**Output**
- Plots: r2_StartingPos.png and r2_EndingPos.png (global Predicted vs Real)
- Plots for Starting and Ending Position (Cases 1-4) (global Predicted vs Real)

### Trimming
Previously, we will need to generate the dataset from the FASTA files with the sequences that we want to trim. 
This will generate a .txt file where the trimmed sequences will be saved with the original header name (until # symbol) and indicates the positions in which the sequence was cut.

**Batch processing (HPC/SLURM):**
```batch
sbatch auto_trimming.sh
```
**Executes** `Auto_trimming.py with --mode trimming`

**Parameters**
- `--input_fasta`: Path to FASTA file from which the dataset was created (required)
- `--dataset_dir`: Directory where the dataset is saved
- `--model`:
- `--scaler`:

**Output**
- curated_seq.txt
