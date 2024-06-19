# Analysis and Detection of Face Morphing using Graph Neural Networks.
Bachelor's degree thesis project Repository

## Prerequisites

- Python 3.11 or later
## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/GiuseppeBas98/Progetto_Tesi.git
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Extract** the Datasets from the .rar file named: datasets.rar
2. **Build** the Dataloaders from the Datasets:
   1. Run "CreateSMDDTrainSubFolders.py" located in Windows Folder -> Build Folder
   2. Run "RunAllDataMORPHEDTrain.py" and "RunAllDataBONAFIDETrain.py" located in Windows Folder -> Build Folder -> CreateDataTrain Folder
   3. Run "CreateAllDataLoaders.py" located in Windows Folder -> Build Folder -> CreateDataTest Folder
3. **Start** Model's learning: Run "CreateBinaryModelWITHCUDA.py" located in Windows Folder -> Usage Folder
4. **Use** Model: Run "useTrainedModel.py" located in Windows Folder -> Usage Folder
5. **Reminder** There are specific folders where dataloaders and models are saved. 






