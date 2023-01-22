Thank you for viewing the code.

First, install packages for Python 3.7:
cd code
pip install -r requirements.txt

Next, the data is formatted and a existence probability map is generated:
python track_to_hdf5.py
python generator_EP.py

Learning 3D U-Net:
python main_backbone.py

You can change to other settings by editing config/main_backbone.yaml.

Learning Graph Transformer:
python main.py

Similarly, you can change to other settings in Graph Transformer by editing config/main.yaml.


Finally, the learned model is tested:
python test.py