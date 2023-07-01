# EyeQ-Foundation

EyeQ-Foundation is a repository contains some zero shot computer vision models. The aim of this project is to make process automatic. For example automatic label service.

Funcationalities:

- Zero Shot Object Detection
- Auto Label image from directory
- REST API service
- Pip installation
- Docker installatin for REST Server using FastAPI
- CLIP to correctly identify the labels


## Available Models:

- Segment Anything
- Grounding Dino
 
## TODO:

- [ ] Segment Anything HQ
- [ ] Batched Inference
- [ ] RAM
- [ ] Tag2Text
- [ ] Knowledge Distillation application
- [ ] CLIP 


## How to use

### Pip Package:

**Installation:**

```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
pip install git+https://github.com/openai/CLIP.git
```

**Usage in code:**

```
from eyeq_foundation import SAM, Dino
teacher_model = SAM()
```


### Docker Service:

**Build :**
```
docker build -t eyeq-foundation -f docker/Dockerfile-cpu .
```

**Start:**

```
docker run -d --name eyeq-foundation -p 9000:9000 eyeq-foundation
```

**Logs:**

```
docker container logs eyeq-foundation
```


