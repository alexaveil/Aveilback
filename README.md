# GPT3 Chatbot

Code for training, doing inference, deploying and running a conversational chatbot.

## Dependencies
To create a conda env with required dependencies run:

```bash
conda env create --name envname --file=environments.yml
```
## Running server
To run the server, make sure to copy the config example and replace the values for the correct ones.
Also make sure the certificates are in place. Then run:
```python
python server.py [--run_local] [--debug]
```
## Training T5 model
To run the training process on a dataset with the persona chat dataset format, detached run the following:
```bash
nohup python -u transformers_pytorch/train.py
--model_checkpoint google/t5-v1_1-base #(Which initial model to run)
--train_batch_size 1 #(Train batch size)
--valid_batch_size 1 #(Validation batch size)
--n_epochs 10 #(Epochs to run the model for)
--dataset_path dataset/personachat_self_original.json #(Path to the dataset json)
--dataset_cache dataset/personachat_self_original.json_cache #(Path where the dataset cache will be)
--max_history 5 #(How many steps to use in the conversation for training, the more the better, but will require more gpu memory)
--num_candidates 2 #(How many response candidate to use per message)
--gradient_accumulation_steps 16 #(After how many batches update the model)
> out.out &
```
## Training evaluation
To evaluate the training of the model, you can use tensorboard. To run the tensorboard logger run:
```bash
bash transformers_pytorch/tensorboard.sh
```
Make sure that the logdir has the correct path.
Then go to **localhost:6006** to see the progress.
If you want to get the metrics for a model after the training ended, you can use the training script to run evaluation before the training only, generating the wanted metric values.
## Inference
For testing doing inference run the following script where $PATH_TO_MODEL_CHECKPOINT_FOLDER is where you have the trained model. Make sure that if there isn't a **pytorch_model.bin** file to rename the copy the last checkpoint to that filename.
```bash
python -u transformers_pytorch/inference.py --checkpoint_path $PATH_TO_MODEL_CHECKPOINT_FOLDER
```
## Deploying model to Cloud Run
Follow instructions in deploy folder readme.
