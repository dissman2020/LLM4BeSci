## LLM4BeSci

This repository contains the code and data for the paper ["A tutorial on open-source large language models for 
behavioral science"](https://doi.org/10.3758/s13428-024-02455-8) by Zak Hussain, Marcel Binz, Rui Mata, and Dirk U. Wulff.

### Instructions for running the code in Google Colab
1. If you do not have a Google account, you will need to create one.
2. Navigate to Google Drive (https://drive.google.com/).
3. In the top-left, click New > More > Colaboratory. If you do not see Colaboratory, you may need to click "Connect more apps",
   search for 'Colaboratory', and install it. Then click New > More > Colaboratory.
4. Run the following code snipped in the first cell (```shift + enter```) of your notebook to mount your Google Drive to the Colab environment.
   A pop-up will ask you to connect, click through the steps to connect your Google Drive to Colab (you will have to do this
   every time you open a new notebook).
```
from google.colab import drive
drive.mount("/content/drive")
```
5. Clone the GitHub repository to your Google Drive by running the following code snippet in the second cell of your notebook:
```
%cd /content/drive/MyDrive
!git clone https://github.com/Zak-Hussain/LLM4BeSci.git
```
6. Go back to your Google Drive and navigate to the folder ```LLLM4BeSci```. You should see the directories 
```choice```, ```crt```, ```health```, and ```personality``` containing the relevant notebooks (.ipynb files) and data
   (it may take a few minutes for the files to appear).
7. The notebooks are designed to be run with the GPU enabled. To do this, click Runtime > Change runtime type > 
Hardware accelerator > T4 GPU.
8. Run the first cell of the notebook to install the required packages. This may take a few minutes and ask for you to
   give permission to access your Google Drive.
   You are now ready to start the exercises!

### Citing this work
APA citation: 

Hussain, Z., Binz, M., Mata, R., & Wulff, D. U. (2024). A tutorial on open-source large language models for behavioral science. *Behavior Research Methods*, 1-24.

Bibtex citation:

```
@article{hussain2024tutorial,
  title={A tutorial on open-source large language models for behavioral science},
  author={Hussain, Zak and Binz, Marcel and Mata, Rui and Wulff, Dirk U},
  journal={Behavior Research Methods},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```
