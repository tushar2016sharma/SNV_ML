# SNV_ML
### Modeling the origin of the mutation type in scRNA nanopore long reads data.

Installation requirements - all the dependencies are listed in the requirements.txt file. 

To run the models either in an existing or a new environment, use the following snippet:

```
# Clone the repository
git clone https://github.com/tushar2016sharma/SNV_ML.git
cd SNV_ML

# Install the dependencies
!pip install -r requirements.txt

# Check GPU availability
import torch
print("Using device:", "GPU" if torch.cuda.is_available() else "CPU")

# Run the model
python Models/FNN_dna_rna.py   # or CNN_dna_rna.py for the CNN model
```
