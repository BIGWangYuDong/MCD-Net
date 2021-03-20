install environment based on (mmdetection)[]
pip install -r requirements.txt

put the test data in DATA/Test/train and DATA/Test/gt (both need)


rename the repo to "Dehaze"

change root in configs/MCDNet.py (line 15,16)
change root in test_.py(line48,51)

run test_.py 

the pretrained model can be download at google drive. 
