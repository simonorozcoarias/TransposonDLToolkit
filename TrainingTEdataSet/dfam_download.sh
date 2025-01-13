#!/bin/bash

#SBATCH -o download.out
#SBATCH -e download.err
#SBATCH --mail-type END
#SBATCH --mail-user email@uoc.edu
#SBATCH -J DFam_download
#SBATCH --time 5-23:00:00
#SBATCH --partition long
#SBATCH -n 60
#SBATCH -N 1
#SBATCH --mem 350GB
#SBATCH --export=NONE

# Variable del path hacia el programa FamDB
famdb_path="./FamDB-1.0.1/famdb.py"
# Variable del path hacia la base de datos DFam descargada para FamDB
dfam_dir="./dfam"

# Verifica el parámetro de entrada
mode="${1:-curated}" # Por defecto es "curated" si no se proporciona parámetro

if [[ "$mode" == "curated" ]]; then
    output_dir="./libraries_curated"
    famdb_flag="--curated"
elif [[ "$mode" == "uncurated" ]]; then
    output_dir="./libraries_NO_curated"
    famdb_flag="--uncurated"
elif [[ "$mode" == "all" ]]; then
    output_dir="./libraries_all"
    famdb_flag=""
else
    echo "Error: Parámetro inválido. Usa 'curated', 'uncurated' o 'all'."
    exit 1
fi

species_file="list_species.txt"

# Crear el directorio de salida si no existe
mkdir -p "$output_dir"

# Procesar cada especie en el archivo
while IFS= read -r species; do
    # Crear un nombre de archivo válido para la especie
    file_name=$(echo "$species" | tr ' ' '_' | tr -cd '[:alnum:]_')
    output_file="${output_dir}/${file_name}.fasta"

    # Verificar si el archivo ya existe
    if [[ -f $output_file ]]; then
        echo "File already exists for: $species. Skipping."
        continue
    fi

    # Descargar las secuencias para esa especie de DFam
    echo "Processing: $species"
    "$famdb_path" -i "$dfam_dir" families -f fasta_name $famdb_flag --include-class-in-name "$species" > "$output_file"

    # Comprobar que el fichero no esté vacio
    if [[ -s $output_file ]]; then
        echo "Saved in: $output_file"
    else
        echo "Not generated for: $species. Empty file"
        rm "$output_file"
    fi
done < "$species_file"

echo "Process finished. Archives saved in $output_dir"
