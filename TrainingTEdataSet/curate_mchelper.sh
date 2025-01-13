#!/bin/bash

#SBATCH -o MCH_HV.out
#SBATCH -e MCH_HV.err
#SBATCH --mail-type=END
#SBATCH --mail-user=email@uoc.edu
#SBATCH -J MCH_HV
#SBATCH --time=13-23:59:59
#SBATCH --partition=long
#SBATCH -n 60
#SBATCH -N 1
#SBATCH --mem=350GB
#SBATCH --export=NONE

# Load the conda environment
source activate MCHelper

# Define base path
path="/shared/home/sorozcoarias/tagua_gen_ec/InpactorDB2.0/Jordi"
# List of species to curate
species_file="${path}/sorted_vertebrata.txt"
# Extra log output for runnign the process multiple times
success_log="${path}/success.log"
failure_log="${path}/failure.log"

# Check if the species file exists and is not empty
if [[ ! -s "$species_file" ]]; then
    echo "Error: Species file is empty or missing: $species_file"
    exit 1
fi

# Read the first species from the file
species=$(head -n 1 "$species_file")
sed -i '1d' "$species_file" # Remove the first line to avoid processing the same line

echo "Processing species: $species"

# Convert species name to a safe filename format
file_name=$(echo "$species" | tr ' ' '_' | tr -cd '[:alnum:]_')

# Step 1: Verify necessary files and directories
library_path="${path}/libraries_NO_curated/${file_name}.fasta"
busco_path="${path}/busco/vertebrata_odb10.hmm"
output_dir="${path}/libraries_AUTO_curated/${file_name}"
output_file="${path}/curated_MCHelper/${file_name}.fasta"
genomes="${path}/genomes/${file_name}"

# Where .fasta file should be saved for species
if [[ ! -f "$library_path" ]]; then
    echo "Error: Library file not found for $species: $library_path" >> "$failure_log"
    sbatch $0
    exit 0
fi

# BUSCO file -> could get automated
if [[ ! -f "$busco_path" ]]; then
    echo "Error: BUSCO file not found: $busco_path" >> "$failure_log"
    exit 1
fi

if [[ ! -d "$output_dir" ]]; then
    mkdir -p "$output_dir"
fi

if [[ ! -d "${path}/curated_MCHelper" ]]; then
    mkdir -p "${path}/curated_MCHelper"
fi

if [[ ! -d "$genomes" ]]; then
    mkdir -p "$genomes"
fi

# Step 2: Download the genome dataset
# Requires NCBI datasets library
echo "Downloading genome dataset for ${species}..."
datasets download genome taxon "${species}" --reference --filename "${genomes}/ncbi_dataset.zip"

# Unzip the dataset
echo "Unzipping genome dataset..."
unzip -o "${genomes}/ncbi_dataset.zip" -d "${genomes}/" || {
    echo "Error: Failed to unzip genome dataset for $species" >> "$failure_log"
    sbatch $0
    exit 1
}

# Step 3: Locate the .fna file (the genome file)
echo "Locating .fna file in ${genomes}/..."
fna_file=$(find "${genomes}/" -type f -name "*.fna" -print -quit)

if [[ -z "$fna_file" ]]; then
    echo "Error: No .fna file found for $species" >> "$failure_log"
    sbatch $0
    exit 1
fi

echo "Running MCHelper.py for ${species}" >> "$success_log"

# Step 4: Run the Python script
python "${path}/MCHelper/MCHelper.py" -x 10 -t 60 -l "${library_path}" \
   -o "${output_dir}" --input_type fasta \
   -g "${fna_file}" \
   -b "${busco_path}"

if [[ $? -eq 0 ]]; then
    if [[ ! -f "${output_dir}/classifiedModule/kept_seqs_classified_module_curated.fa" ]]; then
        echo "Error: $species processed but fasta file not found" >> "$failure_log"
        sbatch $0
        exit 1
    fi
    echo "Success: $species processed successfully" >> "$success_log"
    # Save file to curated folder with species name -> could be upgraded to appendix to file with already curated sequences if it exists
    mv "${output_dir}/classifiedModule/kept_seqs_classified_module_curated.fa" "${output_file}"
else
    echo "Error: MCHelper.py failed for $species" >> "$failure_log"
fi

# Step 5: Clean up the genomes directory
rm -rf "${genomes:?}"/*
rm -rf "${output_dir:?}"/*

# Step 6: Recurse if species remain in the file
if [[ -s "$species_file" ]]; then
    echo "Submitting next species..."
    sbatch $0
else
    echo "All species processed."
fi
