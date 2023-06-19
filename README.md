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


### Build Docker:
```
docker build -t eyeq-foundation -f docker/Dockerfile-cpu --progress=plain .
```

### Start Docker:

```
docker build -t eyeq-foundation -f docker/Dockerfile-cpu --progress=plain .
```

### Logs:

```
docker container logs eyeq-foundation
```


