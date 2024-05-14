from datasets import load_dataset

train_data =load_dataset('wikipedia','20220301.simple', split='train')
train_data.to_json('wikipedia.json', lines=True)

