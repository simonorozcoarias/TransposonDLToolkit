# Repository for the dataset created for training the next generation DL algorithms for detecting TEs

## Overview

This repository contains curated and uncurated datasets derived from Dfam, along with scripts for downloading and managing these datasets. The data is intended to train deep learning algorithms to detect transposable elements in genomic sequences.

## Current Data Structure

### Folders:

Folders generated when using the scripts:

- **`libraries_curated`**: Contains curated datasets from Dfam.
- **`libraries_NO_curated`**: Contains uncurated datasets from Dfam.
- **`curated_MCHelper`**: Contains curated datasets using MCHelper.
- **`libraries_all`**: If generated, this folder contains both curated and uncurated datasets.

## Scripts

### Download Sequences from Dfam

#### Requirements

1. **FamDB Tool**  
   Download the `FamDB` program from [Dfam's GitHub repository](https://github.com/Dfam-consortium/FamDB).
2. **Dfam Database**  
   Download the Dfam database into a folder (e.g., `dfam`) from [Dfam's official release page](https://dfam.org/releases/Dfam_3.8/families/FamDB/).
3. **List of Species**  
   Prepare a `.txt` file with a list of scientific names of species. Each line should contain one species name, for example:

```
Scolopax mira
Callorhinchus milii
Limosa lapponica baueri
Bettongia penicillata ogilbyi
Sceloporus occidentalis
Turnix velox
Uria aalge
Uria lomvia
Anguilla anguilla
Alca torda
Pelodiscus sinensis
Cepphus grylle
```

#### Usage Instructions

1. **Preprocessing (Optional but Recommended)**:  
   Before executing the script, run the `dos2unix` command on both the script file and the species list to ensure compatibility:

```bash
dos2unix ./dfam_download.sh list_species.txt
```

2. **Execution**:
   Run the download script with the following command:

```bash
./dfam_download.sh
```

Alternatively, use `sbatch` to queue the script:

```bash
sbatch ./dfam_download.sh
```

3. **Parameters**:
   The script accepts three optional parameters:

    - `curated`: Downloads only curated datasets and saves them in the `libraries_curated` folder.
    - `uncurated`: Downloads only uncurated datasets and saves them in the `libraries_NO_curated` folder.
    - `all`: Downloads both curated and uncurated datasets, saving them in the `libraries_all` folder.

4. **Additional Options**:
   To include ancestral sequences in the downloaded data, add the `-a` parameter to the `famdb.py` command within the script.

---

### Curate sequences using MCHelper

#### Requirements

1. **MCHelper Tool**  
   Download the `MCHelper` program from [MCHelper's GitHub repository](https://github.com/GonzalezLab/MCHelper).
2. **Uncurated Sequences**  
   Inside the folder `libraries_NO_curated`, the `.fasta` files you want to curate (like the ones downloaded from DFam).
3. **BUSCO Lineages**  
   Obtain the BUSCO lineage sequences from [BUSCO's official repository](https://busco-data.ezlab.org/v5/data/lineages/).
4. **List of Species**  
   Prepare a `.txt` file with a list of scientific names of species. Each line should contain one species name, for example:

```
Scolopax mira
Callorhinchus milii
Limosa lapponica baueri
Bettongia penicillata ogilbyi
Sceloporus occidentalis
Turnix velox
Uria aalge
Uria lomvia
Anguilla anguilla
Alca torda
Pelodiscus sinensis
Cepphus grylle
```

#### Usage Instructions

1. **Preprocessing (Optional but Recommended)**:  
   Before executing the script, run the `dos2unix` command on both the script file and the species list to ensure compatibility:

    ```bash
    dos2unix ./curate_mchelper.sh list_species.txt
    ```

2. **Execution**:
   Run the download script with the following command, using `sbatch` to queue the script:

    ```bash
    sbatch ./dfam_download.sh
    ```

    The program can be queued multiple times, without curating the same data or having conflicts.

1. **Parameters**:
   This execution doesn't need aditional parameters, but might be relevant to tweak MCHelper's call based on the computer's capacity
