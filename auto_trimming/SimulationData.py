from Bio import SeqIO
import random
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  
ruta_fasta = filedialog.askopenfilename(title="Selecciona un archivo FASTA", filetypes=[("FASTA files", "*.fasta"), ("All files", "*.*")])

if not ruta_fasta:
    print("No se seleccionó ningún archivo. Saliendo...")
    exit()

# cadena de ADN aleatoria con la longitud deseada
def generar_cadena(longitud):
    bases = ['A', 'T', 'C', 'G']
    return ''.join(random.choices(bases, k=longitud))

# formato FASTA
def datos_entrenamiento(caso):
    total_length = 20000
    sequences = []
    
    with open(ruta_fasta, "r") as fasta_file:
        sequences = list(SeqIO.parse(fasta_file, "fasta"))

    if not sequences:
        print("Error: No se encontraron secuencias en el archivo FASTA.")
        exit()

    if caso == 1:
        random_sequence1 = random.choice(sequences)
        starting_pos_1 = random.randint(0, total_length // 2)
        min_starting_pos_1 = min(len(random_sequence1.seq), total_length // 2)
        starting_pos_1 = random.randint(min_starting_pos_1, total_length // 2)
        random_sequence2 = random.choice(sequences)
        min_starting_pos_2 = min(starting_pos_1 + len(random_sequence1.seq), total_length - len(random_sequence1.seq))
        starting_pos_2 = random.randint(min_starting_pos_2, total_length - int(total_length * 0.1))
        
        final_seq = generar_cadena(starting_pos_1) + random_sequence1.seq
        final_seq += generar_cadena(starting_pos_2 - len(random_sequence1.seq))
        final_seq += random_sequence2.seq

        if len(final_seq) < total_length:
            final_seq += generar_cadena(total_length - len(final_seq))

        final_seq = final_seq[:total_length]
        return f">Caso{caso}_{random_sequence1.id}_{starting_pos_1}_{len(random_sequence1.seq)}\n{final_seq}"

    elif caso == 2:
        sequence_1, sequence_2 = "", ""

        while len(sequence_1) < total_length // 3:
            sequence_1 += str(random.choice(sequences).seq)

        while len(sequence_2) < total_length // 3:
            sequence_2 += str(random.choice(sequences).seq)

        seq_final = sequence_1 + sequence_2 + sequence_1
        missing = int((total_length - len(seq_final)) / 2) + 1

        if len(seq_final) < total_length:
            seq_final = generar_cadena(missing) + seq_final + generar_cadena(missing)

        seq_final = seq_final[:total_length]
        return f">Caso{caso}_{random.choice(sequences).id}_X_{len(sequence_2)}\n{seq_final}"

    elif caso == 3:
        monomer_length = random.randint(5, 100)
        monomer = generar_cadena(monomer_length)
        random_sequence = random.choice(sequences)
        start_pos = random.randint(int(total_length * 0.4), int(total_length * 0.6))
        sequence_length = len(random_sequence.seq)

        microsatelite = (monomer * (start_pos // monomer_length))[:start_pos]
        microsatelite += random_sequence.seq
        microsatelite += (monomer * ((total_length - len(microsatelite)) // monomer_length))[:total_length - len(microsatelite)]

        return f">Caso{caso}_{random_sequence.id}_{start_pos}_{sequence_length}\n{microsatelite}"

# Output
nombre_salida = "output.fasta"
with open(nombre_salida, "w") as archivo_salida:
    for caso in range(1, 3001):
        for _ in range(3):
            archivo_salida.write(datos_entrenamiento(caso) + "\n")

print(f"Archivo generado: {nombre_salida}")
