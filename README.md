# Introduction to NLP
## Assignment-1
### Akshit Sharma (2021101029)

### tokenizer.py
#### Assumptions for tokenizer:
- mentions can be alphanumeric and may contain '_' symbol
- URLs can start with http,https or www
- Removed the dots following Mr, Ms, Mrs and Dr to svoid splitting as sentences.
- Splitting sentences by ./?/! only
- Not removing other punctuations and making them tokens.
- Two new line characters occuring together also used to split as sentences.
- For the tokenizer, the file is run by the command given below. It expects a string as input and returns the split sentences as list of lists (of strings) as output.
 > python3 tokenizer.py

### language_model.py
#### Commmands used to get different outputs for the LMs:

- For the 3-gram model without smoothing, we run using the command given below. It takes a sentence as input and returns its probability score.
 > python3 language_model.py n <corpus_path>

- For the 3-gram model + Good-Turing Smoothing, the command to get perplexity score for an input sentence is:
 > python3 language_model.py g <corpus_path>

- Assumption: The new Z_r values are taken from the regression line itself for each r (no cutoff chosen to take values from line).

- For the 3-gram model + Good-Turing Smoothing, the command to get average perplexity on test set is:
 > python3 language_model.py g <corpus_path> p_test

- To get the average perplexity for train set, command is:
 > python3 language_model.py g <corpus_path> p_train

- To print the average perplexity for test set follwed by its each sentence with its perplexity score, the command is:
 > python3 language_model.py g <corpus_path> file_test

- The same for train set has the command:
 > python3 language_model.py g <corpus_path> file_train

- The same functionality for 3-gram + Linear Interpolation can be obtained by replacing g by i in each of the above commands.

### Generator.py
- Generation can be done for normal N-gram model (any N) without smoothing as well as for one with linear interpolation.

#### Commands used for Generation :

- For N-gram model with linear interpolation, the command is given below. It accepts an input sentence and prints the k most probable words (with probability) theat can follow. 
 > python3 generator.py i <corpus_path> k 

- For N-gram model without interpolation, command is given below. It accepts an input sentence and prints the k most probable words (with probability) theat can follow. The <n> is the N for the model (given as input, if not then 3 gram used). Assumption: If context is unseen, most probable k words are printed based on frequency of individual occurence.
 > python3 generator.py i <corpus_path> k <n>

- For out of data scenerio with a given n for N-gram model, command is given below. Here n is the n for N-gram and q is the number of words to be generated in the sentence.
 > python3 generator.py r <corpus_path> <n> <q>