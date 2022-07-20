docker build -t mirrors.tencent.com/weiyu_docker/moco_flow:latest ./
docker push mirrors.tencent.com/weiyu_docker/moco_flow:latest
docker run --gpus all -v /apdcephfs:/apdcephfs -it mirrors.tencent.com/weiyu_docker/moco_flow:latest /bin/bash
