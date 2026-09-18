from load_data import dataset


data = dataset["train"]


    
def clean(example):
    
    question = str(example["question"]).strip()
    answer =  str(example["answer"]).strip()

    return{
        "question": question,
        "answer":   answer
    }  
    
    


clean_examples = []

for example in data :
    
    clean_example = clean(example)
    
    if clean_example["question"] != "" and clean_example["answer"] != "":
        clean_examples.append(clean_example)

print("cleaned_example",clean_examples[0])
        
    
    