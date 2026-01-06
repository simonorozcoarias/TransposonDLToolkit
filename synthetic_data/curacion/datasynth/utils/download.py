import os
from Bio import SeqIO
from Bio import Entrez



def download_microsatellites_from_ncbi(output_file, max_records=5000):
    """
    Descarga microsatélites desde NCBI
    """

    def load_and_convert(output_file):
        microsats_sequences = []
        try:
            with open(output_file, "r") as f:
                sequences = list(SeqIO.parse(f, "fasta"))
                for seq in sequences:
                    seq.seq = seq.seq.lower().replace('n', '')
                    microsats_sequences.append(seq)
            return microsats_sequences
        except Exception as e:
            print(f"Error cargando el archivo de microsatélites: {e}")
            return []

    if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
        print(f"-  Cargando microsatélites desde {output_file}")
        return load_and_convert(output_file)

    query = f'microsatellite[Title] OR "short tandem repeat"[Title] OR "STR"[Title]'
    print(f"-  Descargando {max_records} microsatélites desde NCBI...")
    try:
        handle = Entrez.esearch(db="nucleotide",
                                term=query,
                                retmax=max_records)
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        print(f"   Encontrados {len(id_list)} registros.")

        if not id_list:
            return None

        handle = Entrez.efetch(db="nucleotide",
                               id=id_list,
                               rettype="fasta",
                               retmode="text")
        sequences = handle.read()
        handle.close()

        with open(output_file, "w") as f:
            f.write(sequences)

        return load_and_convert(output_file)

    except Exception as e:
        print(f" !!!!! Error en la descarga de microsatélites: {e}")
        return None


def download_genes_from_ncbi(output_file, max_records=5000):
    """
    Descarga secuencias de genes de copia múltiple (e.g., rDNA) desde NCBI
    para usarlas en el Caso 5 (Falsos Positivos).
    """

    def load_and_convert(output_file):
        gene_sequences = []
        try:
            with open(output_file, "r") as f:
                sequences = list(SeqIO.parse(f, "fasta"))
                for seq in sequences:
                    seq.seq = seq.seq.lower().replace('n', '')
                    gene_sequences.append(seq)
            return gene_sequences
        except Exception as e:
            print(f"Error cargando el archivo de genes: {e}")
            return []

    if os.path.exists(output_file) and os.path.getsize(output_file) > 100:
        print(f"-  Cargando genes de copia múltiple desde {output_file}")
        return load_and_convert(output_file)

    # Buscamos elementos típicamente de copia múltiple (rDNA, Histonas)
    query = f'("ribosomal DNA"[Title] OR "rRNA gene"[Title] OR "histone cluster"[Title]) AND complete[Title]'
    print(
        f"-  Descargando {max_records} genes de copia múltiple desde NCBI...")
    try:
        handle = Entrez.esearch(db="nucleotide",
                                term=query,
                                retmax=max_records)
        record = Entrez.read(handle)
        handle.close()

        id_list = record["IdList"]
        print(f"   Encontrados {len(id_list)} registros.")

        if not id_list:
            return None

        handle = Entrez.efetch(db="nucleotide",
                               id=id_list,
                               rettype="fasta",
                               retmode="text")
        sequences = handle.read()
        handle.close()

        with open(output_file, "w") as f:
            f.write(sequences)

        return load_and_convert(output_file)

    except Exception as e:
        print(f" !!!!! Error en la descarga de genes de copia múltiple: {e}")
        return None
