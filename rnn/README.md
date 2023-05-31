# Recurrent Neural Network based Language Models

This code trains a recurrent neural network based language model and evaluates its performance. It is able to train a model from scratch by using given hyperparameter values, which can be set by changing the variables defined in the "initialize_hyper_parameters" function. When this is done, the final model is saved into a folder called "my_model" which can be found in the same directory after the execution. In addition, it can directly evaluate an existing model by reading it without requiring any training. Deciding to train a model from scratch or using an existing model is simple. "WILL_DIRECTLY_EVALUATE" variable can be set to True or False in order to adjust this behavior. If it uses an existing model, evaluation is made on test set; otherwise, it is done on the validation set. 

## How to run the codes for the project
- First of all, the programming language used in the code is Python and the version used is Python 3.8
- Moreover, the program uses nucleotide sequences of human genes datasets. Therefore, all four of the datasets ought to be downloaded from [nucleotide sequences of human genes datasets](https://drive.google.com/drive/folders/1bJHrZ0v36Om_bY3-nOkuHkKfT_dtJDuN?usp=share_link) and put into the folder called "src".  
- Furthermore, the program uses some external libraries which are listed in "requirements.txt" file inside "src" folder. They ought to be installed. Doing it is actually simple. Start a terminal from "src" folder. Then run "pip install -r requirements.txt" to install them.  
- Lastly, you can run the code by using "python main.py" command after setting the desired hyperparameters and running mode by following the instructions mentioned.

**Note:** The execution time for the code is in terms of hours.  

Musa Nuri İhtiyar
