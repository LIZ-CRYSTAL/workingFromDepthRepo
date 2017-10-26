import os.path

with open('train.csv') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

for line in content:
    b = line.split(",")
    if not os.path.isfile(b[0]):
        print b[0] + ',' + b[1]
    
