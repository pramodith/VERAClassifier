import json

def create_dataset(filename):
    x_dict=[]
    with open(filename,'r') as f:
        text=f.read()
    text=text.split(".")
    for line in text:
        tmp={}
        tmp['x']=line
        tmp['y']="other"
        x_dict.append(tmp)
    with open("datasets/cheetahs.json",'w') as f:
        json.dump(x_dict,f,indent=3)

create_dataset("Wikipedia/cheetahs.txt")