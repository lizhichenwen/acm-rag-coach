.PHONY: help install init ingest index crawl eval calibrate serve ui test lint docker

help:
	@echo "install  : 安装依赖"
	@echo "init     : 建表 + 初始化管理员"
	@echo "ingest   : 导入 data/ 下的语料"
	@echo "index    : 建 BM25 + 向量索引（可用 MAX=1500 分批）"
	@echo "crawl    : 抓取 CF/AtCoder 元数据（CF=6000 AT=2500）"
	@echo "eval     : 四组消融评测"
	@echo "calibrate: 标定相关性阈值"
	@echo "serve    : 启动 API (8000)"
	@echo "ui       : 启动前端 (8501)"
	@echo "test     : 跑测试"

install:
	pip install -r requirements-dev.txt

init:
	python -m acm_coach.cli init-db

ingest:
	python -m acm_coach.cli ingest

index:
	python -m acm_coach.cli index --max-vectors $(or $(MAX),0)

crawl:
	python -m acm_coach.cli crawl --cf-limit $(or $(CF),6000) --at-limit $(or $(AT),2500)

eval:
	python -m acm_coach.cli eval --configs g1,g2,g3,g4 --max-queries $(or $(N),150)

calibrate:
	python -m acm_coach.cli calibrate --sample 120

serve:
	python -m acm_coach.cli serve --reload

ui:
	streamlit run frontend/app.py

test:
	python -m pytest tests/ -v

lint:
	ruff check src tests

docker:
	docker compose up --build
