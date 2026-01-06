# -*- coding: utf-8 -*-
import subprocess
import zipfile
import os
import re
import argparse
import multiprocessing
from functools import partial
import shutil
from Bio import SeqIO

# Function that downloads a specific genome
def download_genome(species_genome, zip_file, env_name="MC_helper_agp"):
    try:
        # First try with --reference
        print(f"Trying downloading {species_genome} with --reference...")

        subprocess.run(["conda", "run", "-n", env_name, "datasets", "download", 
            "genome", "taxon", species_genome, "--reference", "--filename", zip_file],
            check=True, capture_output=True, text=True
        )

        print(f"Download with --reference completed for {species_genome}.")

    except subprocess.CalledProcessError as e:
        try:

            # Second try without --reference
            print(f"Error with --reference. Retrying without --reference for {species_genome}...")

            subprocess.run(["conda", "run", "-n", env_name, "datasets", "download", 
                "genome", "taxon", species_genome, "--filename", zip_file],
                check=True, capture_output=True, text=True
            )

            print(f"Download without --reference completed for {species_genome}.")

        except subprocess.CalledProcessError:
            # If also fails, it will write it on the log file
            with open(failure_log, "a") as f:
                f.write(f"Error: Download not possible for {species_genome}\n")
            print(f"Error: Download not possible for {species_genome}. Registered in {failure_log}.")
            print(f"STDERR:\n{e.stderr}")
            return False

    return True

# Function that unzips a genome .zip file 
def unzip_genome(zip_file, genome_dir):

    if not os.path.exists(zip_file):
        print(f"Error: File {zip_file} was not found")
        return False

    try:
        # Create output folder it if doesnt exist
        os.makedirs(genome_dir, exist_ok=True)
        
        # Unzip the file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(genome_dir)
        
        print(f"File {zip_file} was unzipped in {genome_dir}")
        return True

    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid .zip file or it is corrupted.")
        return False

    except Exception as e:
        print(f"Unexpected error while unzipping {zip_file}: {e}")
        return False

# Function that saves the path of the .fna genome file
def find_fna_file(genome_dir, species_name):

    fna_file = None

    if not os.path.exists(genome_dir):
        print(f"Error: genome directory {genome_dir} does not exist")
        return None

    # Search recursively the first .fna file
    for root, dirs, files in os.walk(genome_dir):
        for file in files:
            if file.endswith(".fna"):
                fna_file = os.path.join(root, file)
                break
        if fna_file:
            break

    if fna_file:
        print(f"Genome of {species_name} downloaded and unzipped.")
        print(f"File .fna found in: {fna_file}")
        return fna_file

    else:
        with open(failure_log, "a") as f:
            f.write(f"Error: File .fna for {species_name} couldnt be found.\n")
        print(f"Error: File .fna for {species_name} couldnt be found. Registered in {failure_log}.")
        return None

# Function to get fasta header and run TE-Aid
def run_extract_and_teaid(header, te_fasta, fna_file, output_dir, env_name="te_aid"):

    if not os.path.isfile("./TE-Aid"):
        print("TE-Aid no encontrado o no ejecutable")
        return False

    if not os.path.isfile(fna_file):
        print(f"Archivo genomico no encontrado: {fna_file}")
        return False

    # Verify if TE fasta is not empty and exists
    if not os.path.exists(te_fasta) or os.path.getsize(te_fasta) == 0:
        print(f"Secuencia no encontrada para {header}")
        return False

    # Execute TE-Aid
    print(f"Ejecutando TE-Aid con {te_fasta} y {fna_file}...")
    try:
        result = subprocess.run(
            f"source ~/anaconda3/etc/profile.d/conda.sh && conda activate {env_name} && ./TE-Aid -q {os.path.abspath(te_fasta)} -g {os.path.abspath(fna_file)} -o {os.path.abspath(output_dir)}",
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash"
        )

        print(f"TE-Aid ejecutado correctamente. Resultados en: {output_dir}")

    except subprocess.CalledProcessError:
        print(f"Error inesperado al ejecutar TE-Aid: {e}")
        return False

    return True

# Create species dictionary with indexes     
def create_species_dict_from_fasta(input_fasta):
    species_dict = {}

    # Get headers of sequences
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    headers = [f">{sequence.description}" for sequence in sequences]

    for line_num, header in enumerate(headers, start=0):
        match = re.search(r'([A-Z][a-z]+_[a-z]+)', header)

        if match:
            species_name = match.group(1)
            species_dict.setdefault(species_name, []).append(line_num)

    print(f"Se detectaron {len(species_dict)} especies unicas en {input_fasta}")
    return species_dict

def process_species(species, sequences, positions, headers, input_fasta, output_dir="./te-aid", genomes_dir="./genomes", env_name="MC_helper_agp"):

    os.makedirs(genomes_dir, exist_ok=True)

    print(f"Especie detectada: {species}")
    species_safe = species.replace("_", " ")
    genome_dir = os.path.abspath(os.path.join(genomes_dir, f"{species}_genome"))
    zip_file = os.path.abspath(os.path.join(genomes_dir, f"{species}.zip"))   

    for position in positions:
        try:
            header = headers[position - 1]
            match_case = re.match(r'^>?([^#\s]+)', header)
            case_name = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())

            # Create output directory for the case
            case_dir = os.path.join(output_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)
            
            new_pdf = os.path.join(output_dir, f"{case_name}.pdf")

            # TE FASTA file 
            te_fasta = os.path.join(case_dir, f"{case_name}.fasta")

            with open(te_fasta, "w") as f:
                f.write(f"{header}\n{sequences[position].seq}\n")

            # Check if pdf exists
            if os.path.exists(new_pdf):
                print(f"PDF ya existe: {new_pdf}")
                shutil.rmtree(case_dir)
                continue

            if not os.path.exists(zip_file):
                download_genome(species_safe, zip_file)
            
            if not os.path.exists(genome_dir):
                unzip_genome(zip_file, genome_dir)

            fna_file = find_fna_file(genome_dir, species)
            if not fna_file:
                print(f"No se encontro archivo .fna para {species}")
                continue

            # Execute TE-Aid
            run_extract_and_teaid(header, te_fasta, fna_file, case_dir, env_name="te_aid")

            original_pdf = os.path.join(case_dir, f"{case_name}.fasta.c2g.pdf")

            if os.path.exists(original_pdf):
                os.rename(original_pdf, new_pdf)
                shutil.rmtree(case_dir)
                print(f"PDF renombrado como: {new_pdf}")
            else:
                print(f"No se encontro el PDF esperado: {pdf_original}")

        except Exception as e:
            print(f"ERROR procesando especie {species}: {e}")

    # Removing genome and case 
    if genome_dir and os.path.exists(genome_dir) and new_pdf and os.path.exists(new_pdf):
        shutil.rmtree(genome_dir)
        if zip_file and os.path.exists(zip_file):
            os.remove(zip_file)
        print(f"Genoma {species} eliminado para liberar espacio.")

def generation_multiprocessing(input_fasta, n_processes, output_dir, genomes_dir="./genomes"):

    os.makedirs(genomes_dir, exist_ok=True)
    
    # Get headers of sequences
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    headers = [f">{sequence.description}" for sequence in sequences]

    total = len(headers)
    print(f"Se encontraron {total} secuencias en {input_fasta}")
    
    species_dict = create_species_dict_from_fasta(input_fasta)
    print(f"Se detectaron {len(species_dict)} especies en total.")
    
    print(f"{species_dict}")

    # Crear procesos
    processes = []
    for species, positions in species_dict.items():
        p = multiprocessing.Process(
            target=process_species,
            args=(species, sequences, positions, headers, input_fasta, output_dir, genomes_dir)
        )
        processes.append(p)

    # Ejecutar procesos en batches de n_processes
    for i in range(0, len(processes), n_processes):
        batch = processes[i:i + n_processes]
        
        print(f"Iniciando batch {i // n_processes + 1} de {(len(processes) + n_processes - 1) // n_processes} "
              f"({len(batch)} procesos en paralelo)...")

        for p in batch:
            p.start()

        for p in batch:
            p.join()

        print(f"Batch {i // n_processes + 1} completado.\n")

    print("Procesamiento completo.")     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True, help="Archivo FASTA de libreria")
    parser.add_argument("--processes", type=int, default=20, help="Numero de procesos paralelos")
    parser.add_argument("--output_dir", default="te-aid", help="Directorio de salida")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generation_multiprocessing(
        args.input_fasta,
        args.processes,
        args.output_dir,
    )
    
    db_dir = os.path.abspath("db")

    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
