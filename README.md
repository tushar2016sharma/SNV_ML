# SNV_ML

### Modeling the origin of the mutation type in scRNA nanopore long reads data.

In single-cell RNA sequencing (scRNA-seq) studies, when matched DNA from the same cell is unavailable, confirming SNV origin through direct DNA comparison is not possible. 
Accordingly, being able to infer whether a variant is DNA-derived or RNA-derived solely from transcriptomic reads becomes a practical necessity in most public and clinical datasets.
This project aimed to tackle these challenges by employing feed-forward neural network (FNN) and convolution neural network (CNN) to detect the mutation type of the SNVs. The nanopore 
long reads sequencing data utilized comprised of two primary datasets: variant reads and reference reads, structured in matrix format with SNVs as rows and cellular barcodes as columns. 
SNVs were annotated as DNA or RNA-derived based on predefined germline, somatic, and RNA mutation lists. The data is arranged in a matrix format where a particular entry indicates the 
read counts supported by an SNV for a particular cell barcode. Corresponding to each row, there is a target label in the labels file, indicating the mutation type.

Installation requirements - all the dependencies are listed in the requirements.txt file. 

To run the models either in an existing or a new environment, use the following snippet:

```
# Clone the repository
git clone https://github.com/tushar2016sharma/SNV_ML.git
cd SNV_ML

# Install the dependencies
pip install -r requirements.txt

# Check GPU availability
import torch
print("Using device:", "GPU" if torch.cuda.is_available() else "CPU")

# Run the models
python Models/FNN_dna_rna.py   # or CNN_dna_rna.py for the CNN model
```
