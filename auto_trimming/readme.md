# Auto Trimming
A multi-input deep neural network for trimming out automatically artifacts in transposable element sequences.

## Training
A 2.3 millions parameters neural network was trained with a simulated dataset, to predict the correct starting and ending point of a transposable element in a container sequence. The dataset  is composed of 9,000 sequences of transposable elements (TEs) of 20,000 bp. Three cases of transposable elements with artifacts at the ends, with another chimeric TE or with simple repeats were considered for the generation of the sequences. The sequences of the first case consist of a DNA fragment + first TE + DNA fragment + second TE + DNA fragment. The sequences of the second case consist of a first TE + second TE + repeat of the first TE. The third case sequences consist of a microsatellite that is repeated in tandem by placing the extracted TE at position 10,000 (in the middle), occupying both sides to the ends. All TEs used in this dataset were taken from Dfam, for the species Drosophila melanogaster. This data set is available at: https://doi.org/10.5281/zenodo.13253778

## results

### Training curves:
Loss vs epochs 
<p align="center">
  <img src="https://github.com/simonorozcoarias/TransposonDLToolkit/blob/main/auto_trimming/Train_Curve_los.png" alt="TE Auto Trimming - training" />
</p>

R2 Score vs epochs
<p align="center">
  <img src="https://github.com/simonorozcoarias/TransposonDLToolkit/blob/main/auto_trimming/Train_Curve.png" alt="TE Auto Trimming - training" />
</p>

Starting position predictions in the test datset
<p align="center">
  <img src="https://github.com/simonorozcoarias/TransposonDLToolkit/blob/main/auto_trimming/r2_StartingPos.png" alt="TE Auto Trimming - training" />
</p>

Ending position predictions in the test datset
<p align="center">
  <img src="https://github.com/simonorozcoarias/TransposonDLToolkit/blob/main/auto_trimming/r2_EndingPos.png" alt="TE Auto Trimming - training" />
</p>
