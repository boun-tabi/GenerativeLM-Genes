# Evaluation of Obtained Models for Detecting Mutations on a Synthetic Mutation Dataset

Comparing the performance of models obtained using various methods is crucial. One of the ways for doing it is using a real life task, which is also called extrinsic evaluation. Since generative models model the probability distribution of different examples of the given domain, we though that they ought to be able to discriminate between nucleotide sequences of real and mutated human genes. Firstly, we decided to check whether or not this is the case for synthetically generated mutations. This code firstly generates such a sequence and then looks at the performance of Laplace Smoothing, RNN and Transformer based models. Due to the fact that synthetic generation process contains randomness, a random seed is given as a hyperparameter. Also maximum number of changes that can be made on a sequence and the number of samples we want to obtain are defined as hyperparameters as well. These hyperparameters are defined as "SEED", "NUMBER_OF_MAXIMUM_CHANGES_PER_SAMPLE", "HOW_MANY_NEW_SAMPLES" in the first line of the code. When this is done, the final model is saved into a folder called "my_model" which can be found in the same directory after the execution. For any given such a tuple, the obtained dataset can be found as a ".csv" file under "SyntheticDataset" folder of the "src" folder. At the end, the accuracy of all the mentioned models are listed when perplexity or probability values are used for the decision.

## How to run the codes for the project
- First of all, the programming language used in the code is Python and the version used is Python 3.8
- Moreover, the program uses nucleotide sequences of human genes datasets. Therefore, all of the datasets ought to be downloaded from [nucleotide sequences of human genes datasets](https://drive.google.com/drive/folders/1bJHrZ0v36Om_bY3-nOkuHkKfT_dtJDuN?usp=share_link) and put into the folder called "src".  
- Furthermore, the program uses some external libraries which are listed in "requirements.txt" file inside "src" folder. They ought to be installed. Doing it is actually simple. Start a terminal from "src" folder. Then run "pip install -r requirements.txt" to install them.  
- Lastly, you can run the code by using "python main.py" command after setting the desired hyperparameters and running mode by following the instructions mentioned.

**Note:** The execution time for the code is in terms of hours.  

Musa Nuri İhtiyar
