# -*- coding: utf-8 -*-
import subprocess
import zipfile
import os
import re
import multiprocessing
import shutil
from Bio import SeqIO
import numpy as np

# Downloads a specific genome for a given species
def download_genome(species_genome, zip_file, env_name="autotrim_env"):
    try:
        # Attempts to download using '--reference' option
        print(f"Trying downloading {species_genome} with --reference...")

        # Running the command as in the terminal
        subprocess.run(["conda", "run", "-n", env_name, "datasets", "download", 
            "genome", "taxon", species_genome, "--reference", "--filename", zip_file],
            check=True, capture_output=True, text=True
        )

        print(f"Download with --reference completed for {species_genome}.")

    except subprocess.CalledProcessError as e:
        try:

            # Attempts to download without '--reference' option
            print(f"Error with --reference. Retrying without --reference for {species_genome}...")

            # Running the command as in the terminal
            subprocess.run(["conda", "run", "-n", env_name, "datasets", "download", 
                "genome", "taxon", species_genome, "--filename", zip_file],
                check=True, capture_output=True, text=True
            )

            print(f"Download without --reference completed for {species_genome} {e}.")

        except subprocess.CalledProcessError:
            # Checks if this also fails
            print(f"Error: Download not possible for {species_genome}.")
            print(f"STDERR:\n{e.stderr}")
            return False

    return True

# Unzips a genome .zip file in the genome directory
def unzip_genome(zip_file, genome_dir):

    if not os.path.exists(zip_file):
        print(f"Error: File {zip_file} was not found")
        return False

    try:
        # Create genome folder it if doesnt exist
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

# Saves the path of the .fna genome file
def find_fna_file(genome_dir, species_name):

    fna_file = None
    
    # Checks if genome directory exists
    if not os.path.exists(genome_dir):
        print(f"Error: genome directory {genome_dir} does not exist")
        return None

    # Search recursively for the first .fna file
    for root, dirs, files in os.walk(genome_dir):
        for file in files:
            if file.endswith(".fna"):
                fna_file = os.path.join(root, file)
                break
        if fna_file:
            break

    # If found, returns the path to the fna file
    if fna_file:
        print(f"Genome of {species_name} downloaded and unzipped.")
        print(f"File .fna found in: {fna_file}")
        return fna_file

    else:
        print(f"Error: File .fna for {species_name} couldnt be found.")
        return None

# Gets fasta header and runs TE-Aid
def run_extract_and_teaid(header, te_fasta, fna_file, output_dir, TEAid_dir="TEAid", env_name="autotrim_env"):

    # Directory of TE-Aid program and R scripts (by default, "TEAid")
    TEAid_dir = os.path.abspath(os.path.join(TEAid_dir, "TE-Aid")) 

    # Checks if TE-Aid program exists
    if not os.path.isfile(TEAid_dir):
        print("TE-Aid was not found.")
        return False

    # Checks if FNA file exists
    if not os.path.isfile(fna_file):
        print(f"FNA file not found: {fna_file}")
        return False

    # Verify if TE fasta is not empty and exists
    if not os.path.exists(te_fasta) or os.path.getsize(te_fasta) == 0:
        print(f"Sequence not found for {header}")
        return False

    print("Sequence was extracted succesfully.")

    # Execute TE-Aid
    print(f"Running TE-Aid with TE fasta:{te_fasta} and FNA file:{fna_file}...")
    try:
        result = subprocess.run(
            [
                "conda", "run", "-n", env_name, TEAid_dir,
                "-q", os.path.abspath(te_fasta),
                "-g", os.path.abspath(fna_file),
                "-o", os.path.abspath(output_dir)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
    
        print(f"TE-Aid completed successfully. Results saved in: {output_dir}")
    
    except subprocess.CalledProcessError as e:
        print(f"Unexpected error while running TE-Aid: {e}")
        print("stdout:", e.stdout)
        print("stderr:", e.stderr)
        return False
    
    return True

# Create species dictionary with indexes     
def create_species_dict_from_fasta(input_fasta):

    # Create dictionary to save species
    species_dict = {}

    # Get headers of sequences
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    headers = [f">{sequence.description}" for sequence in sequences]

    # If identifies an species, it saves it on the dictionary
    for line_num, header in enumerate(headers, start=0):
        match = re.search(r'([A-Z][a-z]+_[a-z]+)', header)

        if match:
            species_name = match.group(1)
            species_dict.setdefault(species_name, []).append(line_num)

    return species_dict

# Process an species from the dictionary (this includes downloading genome, running TE-Aid and getting the images)
def process_species(species, sequences, positions, headers, TEAid_dir, output_dir="./te_aid", genomes_dir="./genomes"):

    # Checks if genome directory exists
    os.makedirs(genomes_dir, exist_ok=True)

    print(f"Processing species: {species}")
    species_safe = species.replace("_", " ")
    genome_dir = os.path.abspath(os.path.join(genomes_dir, f"{species}_genome"))
    zip_file = os.path.abspath(os.path.join(genomes_dir, f"{species}.zip"))   

    # Loops through the positions for a species
    for position in positions:
        try:
            header = headers[position]
            match_case = re.match(r'^>?([^#\s]+)', header)
            case_name = match_case.group(1) if match_case else re.sub(r'\W+', '_', header.strip())

            # Create output directory for the case
            case_dir = os.path.join(output_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)
            
            # New name for the PDF file (ends in .pdf)
            new_pdf = os.path.join(output_dir, f"{case_name}.pdf")

            # TE FASTA file 
            te_fasta = os.path.join(case_dir, f"{case_name}.fasta")

            print(f"genome_dir: {genome_dir}")
            print(f"case_dir: {case_dir}")
            print(f"case_name: {case_name}")
            print(f"header: {header}")
            print(f"position: {position}")
            print(f"sequences[position].seq: {sequences[position].seq}")

            # Extracts sequence with matching header and saves it in TE fasta
            with open(te_fasta, "w") as f:
                f.write(f"{header}\n{sequences[position].seq}\n")

            # Check if pdf exists
            if os.path.exists(new_pdf):
                print(f"PDF already exists: {new_pdf}")
                
                # Removes case directory
                shutil.rmtree(case_dir)
                continue

            if not os.path.exists(zip_file):
                download_genome(species_safe, zip_file)
            
            if not os.path.exists(genome_dir):
                unzip_genome(zip_file, genome_dir)

            fna_file = find_fna_file(genome_dir, species)
            if not fna_file:
                print(f"FNA file not found for species {species}")
                continue

            # Execute TE-Aid
            run_extract_and_teaid(header, te_fasta, fna_file, case_dir, TEAid_dir, env_name="autotrim_env")

            original_pdf = os.path.join(case_dir, f"{case_name}.fasta.c2g.pdf")

            # Checks if pdf file with original name exists and renames it
            if os.path.exists(original_pdf):
                os.rename(original_pdf, new_pdf)
                shutil.rmtree(case_dir)
                print(f"PDF renamed as: {new_pdf}")
            else:
                print(f"PDF was not found: {original_pdf}")

        except Exception as e:
            print(f"ERROR processing species:{species}. {e}")

    # Removing genome
    if genome_dir and os.path.exists(genome_dir) and new_pdf and os.path.exists(new_pdf):
        shutil.rmtree(genome_dir)
        if zip_file and os.path.exists(zip_file):
            os.remove(zip_file)
        print(f"Genoma {species} eliminado para liberar espacio.")

# Apply multiprocessing to process several species simultaneously
def generation_multiprocessing(input_fasta, TEAid_dir, n_processes=20, output_dir="./te_aid", genomes_dir="./genomes"):

    os.makedirs(genomes_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get headers of sequences
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    headers = [f">{sequence.description}" for sequence in sequences]

    total = len(headers)
    print(f"{total} sequences were found in {input_fasta}")
    
    species_dict = create_species_dict_from_fasta(input_fasta)
    print(f"A total of {len(species_dict)} unique species were detected.")

    # Create processes
    processes = []
    for species, positions in species_dict.items():
        p = multiprocessing.Process(
            target=process_species,
            args=(species, sequences, positions, headers, TEAid_dir, output_dir, genomes_dir)
        )
        processes.append(p)

    # Run processes in batches of n_processes
    for i in range(0, len(processes), n_processes):
        batch = processes[i:i + n_processes]
        
        print(f"Initiating batch {i // n_processes + 1} of {(len(processes) + n_processes - 1) // n_processes} "
              f"({len(batch)} processes in parallel)...")

        for p in batch:
            p.start()

        for p in batch:
            p.join()

        print(f"Batch {i // n_processes + 1} completed.\n")

    print("Processing was completed.") 

    if os.path.exists(genomes_dir):
        shutil.rmtree(genomes_dir)     

    if os.path.exists("db"):
        shutil.rmtree("db")     

# Reads a fasta file and generates JPEG images for each PDF
def generate_te_images(input_fasta, teaid_dir="./te_aid"):

    import fitz

    # List with FASTA headers 
    TEs = list(SeqIO.parse(input_fasta, "fasta"))
    te_image_info = []

    for TE in TEs:
        TE_name = TE.id.split("#")[0]
        species_match = re.search(r'([A-Z][a-z]+_[a-z]+)$', TE.id)
        species_name = species_match.group(0) if species_match else None
        
        pdf_path = os.path.join(teaid_dir, TE_name + '.pdf')
        if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) <= 4 * 1024:
            print(f"PDF not found for {TE_name}, continuing.")
            continue

        print(f"Generating image for TE_name: {TE_name}")
        try:
            doc = fitz.open(pdf_path)
            image_path = os.path.join(teaid_dir, TE_name + ".fa.c2g.jpeg")

            # Save first page as JPEG (adjust if you want multiple pages)
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(200/72, 200/72))
                pix.save(image_path, 'JPEG')

            te_image_info.append({
                "TE": TE,
                "TE_name": TE_name,
                "species_name": species_name,
                "image_path": image_path
            })
        except Exception as ex:
            print(f"Something went wrong with {TE_name}: {ex}")

    return te_image_info
