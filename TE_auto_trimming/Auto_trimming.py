import os, subprocess, re
import argparse
from dataset_library import generation_multiprocessing, generate_te_images
from model_library import ResNet18, auto_trimming, NDStandardScaler, plot_training_metrics, test_model, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
#print(tf.config.list_physical_devices('GPU'))

# Generates .PDF files and images from an input FASTA file   
def load_data(fasta_path):
    
    # Transform sequences from fasta path into TEAid .pdfs
    generation_multiprocessing(fasta_path, TEAid_dir="./TEAid")
    
    print("PDFs generated successfully.")  
    
    # Generate images from TEAid .pdfs
    generate_te_images(fasta_path)
    
    print("Images generated successfully.") 

# Return a model for image plots
def get_model(input_size=(256, 256, 1), num_classes=128):

    tf.keras.backend.clear_session()
    
    # Create four independent CNN branches for each type of plot
    cnn_div = ResNet18.build(input_size, num_classes)
    cnn_cov = ResNet18.build(input_size, num_classes)
    cnn_dot = ResNet18.build(input_size, num_classes)
    cnn_str = ResNet18.build(input_size, num_classes)
                      
    # Combine outputs of the CNNs to produce the final model
    model = auto_trimming(cnn_div, cnn_cov, cnn_dot, cnn_str)
    print(model.summary())
    
    return(model)

# Train a model with specified callbacks
def run_experiment(model, train_ds, dev_ds, num_epochs, steps_per_epoch, validation_steps):

    # Create "checkpoint" folder to save weights
    checkpoint_dir = "./checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_filepath = os.path.join(checkpoint_dir, "model.weights.h5")

    # Reduce learning rate if val_loss doesn't improve
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.01,
        patience=10,
        verbose=1
    )

    # Save best weights according to val_r2_score
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_r2_score",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    # Training
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint_callback, lr_scheduler, early_stopping],
        verbose=2
    )

    return history, checkpoint_callback

# Builds dataset arrays (features, labels, case/species names) from TE images
def build_dataset_from_images(input_fasta, output_dir, images_dir = "./te_aid", TE_size=15000):

    import cv2
    
    # Save TE ids from input FASTA file
    TEs = {TE.id: TE for TE in SeqIO.parse(input_fasta, "fasta")}
    
    # Make a list with names of the image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpeg")]
    
    # Select good images (size > 0)
    good_images = [f for f in image_files if os.path.getsize(os.path.join(images_dir, f)) > 0]
    
    print(f"Found {len(good_images)} valid images out of {len(image_files)} total.")

    # Initialize Numpy matrices and lists to store the data
    feature_data = np.zeros((len(good_images), 256, 256, 4), dtype=np.uint8)
    labels = np.zeros((len(good_images), 2), dtype=np.float32)
    case_names = []
    species_names = []

    n = 0

    for image_file in good_images:

        image_path = os.path.join(images_dir, image_file)

        # Extract TE_name from image file's name (remove extension)
        TE_name_match = re.match(r"(.+)\.fa\.c2g\.jpeg$", image_file)

        # Check if file exists
        if TE_name_match is None:
            print(f"Skipping unrecognized file: {image_file}")
            continue
            
        TE_name = TE_name_match.group(1)
        
        # Search for TE_name if TE list
        TE = next(
            (v for k, v in TEs.items() if k.startswith(TE_name)),
            None
        )
        
        if TE is None:
            print(f"Warning: TE {TE_name} not found in FASTA")
            continue
        
        # Get species from TE id
        species_match = re.search(r'([A-Z][a-z]+_[a-z]+)$', TE.id)
        species_name = species_match.group(0) if species_match else None

        # Transform image into grayscale
        te_aid_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if te_aid_image is None:
            print(f"ERROR: Could not open image {TE_name}")
            continue

        # Split and resize plots, and save into different channels of feature data array
        feature_data[n, :, :, 0] = cv2.resize(te_aid_image[150:1030, 150:1130], (256, 256))
        feature_data[n, :, :, 1] = cv2.resize(te_aid_image[150:1030, 1340:2320], (256, 256))
        feature_data[n, :, :, 2] = cv2.resize(te_aid_image[1340:2220, 150:1130], (256, 256))
        feature_data[n, :, :, 3] = cv2.resize(te_aid_image[1340:2220, 1340:2320], (256, 256))

        # Starting and end position labels
        start_pos = int(TE.id.split("_")[-4])
        TE_len = int(TE.id.split("_")[-3])
        labels[n, 0] = start_pos / TE_size
        labels[n, 1] = min((start_pos + TE_len) / TE_size, 1)

        # Append case name and species to lists
        case_names.append(TE_name)
        species_names.append(species_name)

        print(f"Processed {TE_name} -> n: {n}")
        n += 1

    # Create output directory if it doesnt exist
    os.makedirs(output_dir, exist_ok=True)

    # Save arrays
    np.save(os.path.join(output_dir, "features_data.npy"), feature_data[:n])
    np.save(os.path.join(output_dir, "labels_data.npy"), labels[:n])
    np.save(os.path.join(output_dir, "case_labels.npy"), np.array(case_names))
    np.save(os.path.join(output_dir, "species_labels.npy"), np.array(species_names))

    print(f"Dataset created with {n} TEs.")
    
# ====================
# MAIN
# ====================   
if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["teaid", "dataset", "train", "test", "trimming"], required=True, help="Modo de ejecucion")
    parser.add_argument("--input_fasta", required=True, help="Archivo FASTA de libreria")
    parser.add_argument("--dataset_dir", help="Directorio del dataset")
    
    parser.add_argument("--model", help="Modelo para generar secuencias cortadas")
    parser.add_argument("--scaler", help="Scaler para generar secuencias cortadas")
            
    parser.add_argument("--TEAid_dir", help="Directorio del programa TEAid y dependencias en R")
    parser.add_argument("--processes", type=int, default=20, help="Numero de procesos paralelos")
    parser.add_argument("--output_dir", default="te_aid", help="Directorio de salida")
    
    parser.add_argument("--dataset_testing", help="Directorio del dataset de testeo")  
    
    args = parser.parse_args()
    
    if args.mode == "teaid":
    
        load_data(args.input_fasta)
        
    if args.mode == "dataset":
    
        build_dataset_from_images(args.input_fasta, args.dataset_dir)
    
    elif args.mode == "train":

        batch_size = 16
        num_epochs = 200
        input_size=(256, 256, 1)
        classes=128
    
        x_str = os.path.join(args.dataset_dir, "features_data.npy")
        y_str = os.path.join(args.dataset_dir, "labels_data.npy")
    
        # Load data using NumPy memory mapping (not loading the full dataset)
        x = np.load(x_str, mmap_mode="r")
        y = np.load(y_str, mmap_mode="r")
    
        print(f"Loaded X shape: {x.shape}")
        print(f"Loaded Y shape: {y.shape}")
    
        # Divide data to create subdatasets for training, test and validation, and save indices
        indices = np.arange(len(y))
        train_idx, test_dev_idx = train_test_split(indices, test_size=0.2, random_state=7)
        dev_idx, test_idx = train_test_split(test_dev_idx, test_size=0.5, random_state=7)  

        # Save scaler
        X_train_for_scaler = np.stack([(x[i].astype(np.float32) / 255.0) for i in train_idx])            
        scalerX = NDStandardScaler().fit(X_train_for_scaler)
        scalerX.save_model("scalerX")
            
        # TensorFlow dataset generator
        def make_dataset(indices, shuffle=False, repeat=False):
            def gen():
                for i in indices:
                    xi = x[i].astype(np.float32) / 255.0
                    xi = scalerX.transform(xi[np.newaxis, ...])[0].astype(np.float16)
                    yield (
                        xi[..., 0:1],
                        xi[..., 1:2],
                        xi[..., 2:3],
                        xi[..., 3:4],
                    ), y[i]
    
            ds = tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    (
                        tf.TensorSpec(input_size, tf.float16),
                        tf.TensorSpec(input_size, tf.float16),
                        tf.TensorSpec(input_size, tf.float16),
                        tf.TensorSpec(input_size, tf.float16),
                    ),
                    tf.TensorSpec((2,), tf.float32),
                )
            )
            
            if shuffle:
                ds = ds.shuffle(512)
                
            if repeat:
                ds = ds.repeat()
                
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
        # Create datasets for training, validation and test
        train_ds = make_dataset(train_idx, shuffle=True, repeat=True)
        dev_ds   = make_dataset(dev_idx)
        test_ds  = make_dataset(test_idx)
    
        # Model for training
        tf.keras.backend.clear_session()    
        model = get_model(input_size, classes)

        steps_per_epoch = len(train_idx) // batch_size
        validation_steps = len(dev_idx) // batch_size

        # Fit model on training data
        history, checkpoints = run_experiment(
            model,
            train_ds,
            dev_ds,
            num_epochs,
            steps_per_epoch,
            validation_steps            
        )
   
        # Plots
        plot_training_metrics(history)           
        
        # Guardar modelo entrenado
        model.save('trained_model.h5')

        # Save data for testing            
        X_test = []
        Y_test = []
            
        for i in test_idx:
            xi = x[i].astype(np.float32) / 255.0
            xi = scalerX.transform(xi[np.newaxis, ...])[0].astype(np.float16)
            
            X_test.append((
                xi[..., 0:1],
                xi[..., 1:2],
                xi[..., 2:3],
                xi[..., 3:4],
            ))
            Y_test.append(y[i])
            
        # Separar canales
        ch0, ch1, ch2, ch3 = zip(*X_test)
        X_test = (
            np.stack(ch0, axis=0),
            np.stack(ch1, axis=0),
            np.stack(ch2, axis=0),
            np.stack(ch3, axis=0),
        )
            
        # Convertir Y_test a array
        Y_test = np.stack(Y_test)
            
        # Comprobar shapes
        for ch in X_test:
            print(ch.shape, ch.dtype)
        print("Y_test shape:", Y_test.shape, Y_test.dtype)
            
        # Save arrays
        np.save("X_test.npy", X_test, allow_pickle=True)
        np.save("Y_test.npy", Y_test)
    
    elif args.mode == "test":
         
        # Load model
        model = tf.keras.models.load_model(
          args.model,
          custom_objects={"r2_score": r2_score},
          compile = False
        )
        
        # Calculate predictions for dataset testing    
        predictions = test_model(args.model, args.scaler, args.dataset_testing)
        
        """
        tf.keras.utils.plot_model(
        model,
        to_file='model_plot.png',
        show_shapes=True,
        show_layer_names=True
        )
        """

        print(predictions)

    elif args.mode == "trimming":

        output_fasta = "curated_seq.txt"

        # Calculate start and end positions for sequences included in the dataset
        predictions = test_model(args.model, args.scaler, args.dataset_dir)
        
        TE_ids = np.load(os.path.join(args.dataset_dir, "case_labels.npy"), allow_pickle=True)

        print(f"Predictions: {predictions}")
        
        # Load original sequences
        sequences = list(SeqIO.parse(args.input_fasta, "fasta"))

        # Create a list to save trimmed TEs
        cut_records = []

        TE_size = 15000
        for i, pred in enumerate(predictions):
            TE_id = TE_ids[i]
            print(TE_id)
            start = int(pred[0] * TE_size)
            print(f"Start position:{start}")
            end = int(pred[1] * TE_size)
            print(f"End position:{end}")

            # Get the SeqRecord with matching TE id
            record = next((rec for rec in sequences if rec.id.startswith(TE_id)), None)

            # Check if TE id matches
            if record is None:
                print(f"TE_id {TE_id} not found in sequences")
                continue

            # Trim the sequence from start to end position
            curated_seq = record.seq[start:end]

            # Create new header for new record and append to list
            new_record = SeqRecord(
                Seq(curated_seq),
                id=TE_id,
                description=f"cut from {start} to {end}"
            )
            cut_records.append(new_record)

        print(f"Saving {len(cut_records)} cut sequences to {output_fasta}...")
        SeqIO.write(cut_records, output_fasta, "fasta")
        print("Done!")
