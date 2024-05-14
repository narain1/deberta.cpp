mkdir models
curl -L https://huggingface.co/microsoft/deberta-v3-base/resolve/main/pytorch_model.bin?download=true -o models/pytorch_model.bin
curl -L https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model?download=true -o models/spm.model
curl -L https://huggingface.co/microsoft/deberta-v3-base/resolve/main/config.json?download=true -o models/config.json
