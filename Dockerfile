# 使用Python 3.12官方镜像作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置Python环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app



# 复制项目文件
COPY requirements.txt .
COPY api.py .
COPY core.py .
COPY index.html .
COPY README.md .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 暴露端口
EXPOSE 9000

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9000"] 