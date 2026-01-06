"""
Script completo
- cuenta las secuencias en el archivo FASTA original
- genera un número similar de secuencias sintéticas
- combina ambos datasets en uno solo
"""

from Bio import SeqIO
import argparse
import os
import sys
from pathlib import Path

from types import SimpleNamespace
from generation import generate_synthetic_datasets
from combine_datasets import combine_datasets as combine_datasets_lib


def count_sequences(fasta_file):
    """Cuenta el número de secuencias en un archivo FASTA"""
    count = 0
    try:
        with open(fasta_file, "r") as f:
            for record in SeqIO.parse(f, "fasta"):
                count += 1
    except Exception as e:
        print(f"Error contando secuencias en {fasta_file}: {e}")
        return None
    return count


def generate(fasta_file,
             num_sequences,
             output_dir,
             processes=10,
             fragmentation_prob=0.3,
             microsat_imperfect_prob=0.15,
             download_microsats=False,
             entrez_email=None,
             ncbi_species=None,
             ncbi_max_records=100,
             total_length=20000,
             microsats_file=None):
    """
    Genera el dataset sintético
    """
    print("Generando dataset sintético...")
    print(f"Objetivo: {num_sequences} secuencias (4 casos × {num_sequences // 4} por caso)")
    print(f"Longitud: {total_length} bp")
    seq_per_case = (num_sequences + 3) // 4

    try:
        generate_synthetic_datasets(entrez_email, fragmentation_prob, microsat_imperfect_prob, fasta_file, seq_per_case, processes, output_dir, total_length)
        synthetic_file = os.path.join(output_dir, "synthetic_dataset_mixed.fasta")
        if os.path.exists(synthetic_file):
            return synthetic_file
        else:
            return None

    except Exception as e:
        print(f"Error generando dataset sintético: {e}")
        return None


def combine_datasets(synthetic_file,
                     original_file,
                     output_file,
                     labels_file=None):
    """
    Combina los datasets usando combine_datasets.py
    """
    try:
        combine_datasets_lib(
            synthetic_file=synthetic_file,
            original_file=original_file,
            output_file=output_file,
            labels_file=labels_file,
            max_original=None,
            balance=True
        )
        return True
    except Exception as e:
        print(f"Error combinando datasets: {e}")
        return False


def args_parser():
    parser = argparse.ArgumentParser(
        description="Genera dataset sintético y lo combina con el original")

    parser.add_argument("--original",
                        required=True,
                        help="Archivo FASTA original con TEs")
    parser.add_argument("--output_dir",
                        required=True,
                        help="Directorio de salida")
    parser.add_argument("--total_length",
                        type=int,
                        default=20000,
                        help="Longitud de las secuencias")
    parser.add_argument("--max_sequences",
                        type=int,
                        default=None,
                        help="Límite de secuencias")
    parser.add_argument("--processes",
                        type=int,
                        default=10,
                        help="Número de procesos")
    parser.add_argument("--fragmentation_prob",
                        type=float,
                        default=0.3,
                        help="Probabilidad de fragmentación")
    parser.add_argument("--imperfect_prob",
                        type=float,
                        default=0.15,
                        help="Probabilidad de imperfección en microsatélites")
    parser.add_argument("--entrez_email",
                        type=str,
                        default=None,
                        help="Email para Entrez (NCBI)")

    return parser.parse_args()


def main():
    args = args_parser()

    # Check microsatelites file
    microsats_file = "datasets/ncbi_microsats.fasta"
    if os.path.exists(microsats_file):
        print(f"Se encontraron microsatélites en {microsats_file}")
        DOWNLOAD_MICROSATS = False
    else:
        print(f"No se encontró {microsats_file}, se descargarán de NCBI")
        DOWNLOAD_MICROSATS = True

    # Check Fasta
    if not os.path.exists(args.original):
        print(
            f"Error: No se encontró el archivo original: {args.original}")
        sys.exit(1)

    print(
        "----------------- DETERMINAR NUMERO SECUENCIAS SINTETICAS ------------------"
    )

    # Contar secuencias
    num_original = count_sequences(args.original)
    if num_original is None:
        sys.exit(1)

    print(f"Encontradas {num_original} secuencias en el archivo original")

    # Determinar cuántas generar (si se especifica un límite, sino usar el total)
    if args.max_sequences and args.max_sequences < num_original:
        num_to_generate = args.max_sequences
    else:
        num_to_generate = num_original

    print(f"Generando {num_to_generate} secuencias sinteticas")

    # outputdir
    os.makedirs(args.output_dir, exist_ok=True)

    print("----------------- GENERAR DATASET SINTETICO ------------------")

    synthetic_file = generate(args.original,
                              num_to_generate,
                              args.output_dir,
                              processes=args.processes,
                              fragmentation_prob=args.fragmentation_prob,
                              microsat_imperfect_prob=args.imperfect_prob,
                              download_microsats=DOWNLOAD_MICROSATS,
                              entrez_email=args.entrez_email,
                              total_length=args.total_length,
                              microsats_file=microsats_file)

    if not synthetic_file or not os.path.exists(synthetic_file):
        print(f"Error: No se pudo generar el dataset sintetico")
        sys.exit(1)

    # Contar secuencias generadas
    num_synthetic = count_sequences(synthetic_file)
    print(f"Generadas {num_synthetic} secuencias sintéticas")

    print(
        "----------------- JUNTAR DATASET SINTETICO Y ORIGINAL ------------------"
    )
    # Combinar datasets
    combined_file = os.path.join(args.output_dir, "combined_dataset.fasta")
    labels_file = os.path.join(args.output_dir, "combined_labels.txt")

    success = combine_datasets(synthetic_file, args.original, combined_file,
                               labels_file)

    if not success:
        print(f"Error: No se pudo combinar los datasets")
        sys.exit(1)

    # coNtar secuencias totales
    num_combined = count_sequences(combined_file)

    # Resumen final
    print("-------------------  FIN ---------------------")
    print(f"Secuencias originales: {num_original}")
    print(f"Secuencias sintéticas generadas: {num_synthetic}")
    print(f"Secuencias en dataset combinado: {num_combined}")
    print(f"Archivos:")
    print(f"Dataset sintético: {synthetic_file}")
    print(f"Dataset combinado: {combined_file}")
    print(f"Etiquetas: {labels_file}")


if __name__ == "__main__":
    main()
