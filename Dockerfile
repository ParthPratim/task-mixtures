FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

WORKDIR /workspace

RUN pip install transformers
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
RUN pip install bitsandbytes
RUN pip install vllm

RUN apt-get -y install apt-utils sudo
RUN git config --global user.name "Parth Pratim Chatterjee"
RUN git config --global user.email "parth27official@gmail.com"
RUN git config --global --add safe.directory '*'

