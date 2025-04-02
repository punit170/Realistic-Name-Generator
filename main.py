import torch
from model_helpers import characterIndexing, NameGenerator, generate_names, display_output
from configparser import ConfigParser

def main():
    # initiate config parser and fetch input values from 'settings.ini'
    config = ConfigParser()
    config.read('settings.ini')

    namePrefix = config.get("input", "namePrefix").lower()
    genCount = int(config.get("input", "genCount"))
    tolerance = float(config.get("model", "modelTolerance"))
    modelPath = config.get("model","modelPath")

    # select device to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create character indices
    char_indices, indices_char = characterIndexing()

    # Instantiate the model
    # params with same values as when model was trained
    input_size = len(char_indices)
    output_size = len(indices_char)
    loaded_model = NameGenerator(input_size, output_size)
    loaded_model.to(device)

    # Load the trained model
    loaded_model.load_state_dict(torch.load(f'{modelPath}', map_location=device))
    loaded_model.eval()

    # Print generated names after training
    names = generate_names(n=genCount, model=loaded_model, namePrefix=namePrefix, char_indices=char_indices, indices_char=indices_char, device=device, tolerance = tolerance)

    display_output(namePrefix, genCount, names)


if __name__ == "__main__":
    main()