FROM nvcr.io/nvidia/pytorch:23.04-py3

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

WORKDIR /workspace

RUN pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers==4.44.0
RUN pip install sentencepiece
RUN pip install datasets 
RUN pip install accelerate 
RUN pip install bitsandbytes 
RUN pip install einops 
RUN pip install sentence-transformers 
RUN pip install InstructorEmbedding 
RUN pip install peft 
RUN pip install trl 
RUN pip install rouge_score 
RUN pip install evaluate 
RUN pip install optimum
RUN pip install dash
RUN pip install ipykernel
RUN pip install -U pip setuptools wheel
RUN pip install wandb
RUN pip install bitsandbytes==0.44.0

RUN apt-get -y install apt-utils sudo
RUN git config --global user.name "Parth Pratim Chatterjee"
RUN git config --global user.email "parth27official@gmail.com"
RUN git config --global --add safe.directory '*'

