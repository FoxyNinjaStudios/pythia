#!/bin/bash
cd /Users/tejaswigowda/Downloads/Sam3D-Objects-MLX
export PYTHONUNBUFFERED=1
exec conda run --no-capture-output -n sam-3d-mlx python -u server.py
