# Neural Network-Based Realistic Name Generator

This project uses a neural network-based approach to generate realistic names starting with a given prefix. The model was trained using an LSTM (Long Short-Term Memory) architecture, which learns character-level sequences and generates authentic names. The project includes functionality to generate realistic names based on the trained model.

## Requirements

- Python 3.x
- PyTorch

> [!NOTE]
> <br>This project used the following packages during LSTM training: 
> - Numpy
> - Pandas
> - TensorFlow
> - Matplotlib

To install all required dependencies, run the following command:

pip install -r requirements.txt

## How to use the project

### 1. Clone the repository
Clone this repository to your local machine:

git clone <repository_url>
cd <repository_name>

### 2. Set up config in `settings.ini`
Configure the settings by editing the `settings.ini` file. In the `config/` dir, you can set the following parameters:

- **namePrefix**: The starting character(s) for generating names (e.g., `a`, `ba`, `jo`).
- **genCount**: The number of names to generate.

Example of `settings.ini`:

[settings]  
namePrefix = a  
genCount = 20

### 3. Run `main.py`

To generate names based on the prefix and count defined in the `settings.ini`, run the following in the terminal from the project directory:

```bash
python src/main.py
```

This will generate and display a list of names in the console.

## Sample Output

### Console Output Example:


<div style="display: flex; justify-content: space-between; align-items: flex-start;">
  <div style="text-align: center">
    <p>If namePrefix = a and count = 20, the output:</p>
    <img src="output/sample_output_1.png" alt="Sample Output 1" width="500" style="height:auto; max-height: 800px;">
  </div>
  <div style="text-align: center;">
    <p>If namePrefix = ma and count = 7, the output:</p>
    <img src="output/sample_output_2.png" alt="Sample Output 2" width="500" style="height:auto; max-height: 800px;">
  </div>
</div>

### Loss vs Epoch Plot:

The following is  a plot for loss vs. epoch during model training. You can find the plot saved as `loss_vs_epoch.png` in the `output/` folder.

![Loss vs Epoch Plot](output/loss_vs_epoch.png)
