conda create -y -n nn-retrieval python=3.9 numpy tqdm pandas
conda activate nn-retrieval
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 faiss-gpu -c pytorch -c nvidia
conda install -y rdkit=2022.03.1 -c conda-forge
pip install transformers hnswlib
