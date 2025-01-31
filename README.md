# GPT2 (124M)
### Key points
- Completed: The model has been implemented and pretrained using 10B Tokens from fineweb-edu.
- Main reference: https://www.youtube.com/watch?v=l8pRSuU81PU&t=2760s

### Files
- train_gpt2.py: GPT2 implementation and training/validation loop.
- Every single important implementation detail is explained in the code as comments.
- GPT2 (124M) - Andrej Karapathy.pdf: Contains a) a general overview of the project, b) my notes, c) an analysis of how text generated changes with time and d) how prompting affects text generated
- Pretraining model outputs.csv: Contain the model ouptuts recorded throughout pretraining. The first column highlights the training step and the second column indicates the sample number within that training step.
- outputs.csv: Model outputs after pretraining.
- log.txt: Contains the change in training loss, validation loss and hellaswag accuracy score with training steps.
- pretrained models: Contains model checkpoints. The training step is indicated in the file name.

