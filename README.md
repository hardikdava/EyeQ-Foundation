# EyeQ-Foundation

EyeQ-Foundation is a repository contains some zero shot computer vision models. The aim of this project is to make process automatic. For example automatic label service.

Funcationalities:

- Zero Shot Object Detection
- Auto Label image from directory
- REST API service
- Pip installation
- Docker installatin for REST Server using FastAPI 


## Available Models:

- Segment Anything
- Grounding Dino
 
## TODO:

- [ ] Segment Anything HQ
- [ ] Batched Inference
- [ ] Knowledge Distillation application


## How to use

### Pip Package:

**Installation:**

```
pip install git+https://github.com/facebookresearch/segment-anything.git
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


