from datasets import load_dataset
 
dataset = load_dataset('csv', data_files='hr_data.csv')
 
# Print the dataset
print(dataset)