import json
import os

input_file = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/21_Neural_Entropy_Router/DeiT_LT_Neural_Router.ipynb"
output_dir = "c:/Users/bhara/OneDrive/Documents/Prof. R. Venkatesh Babu (IISc)/VAL_MASTER/22_Ablation_Phase1"
output_file = os.path.join(output_dir, "Phase1_CIFAR_Sweeps.ipynb")

os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r") as f:
    nb = json.load(f)

# Update the first markdown cell
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "Next-Gen ATF" in "".join(cell["source"]):
        cell["source"] = [
            "# Ablation Phase 1: Seeds & Imbalance Factors\n", 
            "\n", 
            "This notebook is parameterized for rapid ablation testing. Change the `SEED` and `IMB_FACTOR` below to test the robustness of the Neural Entropy Router against statistical variance and dataset severity."
        ]
        break

# Find the Config class and inject the sweep parameters
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "class Config:" in "".join(cell["source"]):
        source = "".join(cell["source"])
        # We will inject the parameters at the top of the cell
        new_source = [
            "# ==========================================\n",
            "# ABLATION CONFIGURATION (CHANGE THESE)\n",
            "# ==========================================\n",
            "GLOBAL_SEED = 42\n",
            "IMBALANCE_FACTOR = 0.02 # 0.10=IF10, 0.02=IF50, 0.01=IF100\n",
            "# ==========================================\n\n",
            "import random\n",
            "import numpy as np\n",
            "import torch\n",
            "random.seed(GLOBAL_SEED)\n",
            "np.random.seed(GLOBAL_SEED)\n",
            "torch.manual_seed(GLOBAL_SEED)\n",
            "if torch.cuda.is_available():\n",
            "    torch.cuda.manual_seed_all(GLOBAL_SEED)\n\n"
        ]
        
        # Modify the Config class to use IMBALANCE_FACTOR
        lines = source.split('\n')
        for line in lines:
            if "imb_factor =" in line:
                new_source.append(f"    imb_factor = IMBALANCE_FACTOR\n")
            else:
                new_source.append(line + '\n')
                
        cell["source"] = new_source
        break

with open(output_file, "w") as f:
    json.dump(nb, f, indent=2)

print("Phase 1 Ablation Notebook generated successfully!")
