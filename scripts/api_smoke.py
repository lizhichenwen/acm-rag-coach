# -*- coding: utf-8 -*-
"""API 冒烟：注册 → 登录 → 会话 → 提问（非流式）→ 统计 → 就绪检查。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from acm_coach.api.app import app

client = TestClient(app)
with client:
    user = f"stu_test_{json.dumps(0)}".replace('"', "")
    r = client.post("/api/v1/auth/register", json={"username": "stu_demo_01", "password": "Passw0rd123"})
    print("register:", r.status_code, r.json().get("message", r.json()))

    r = client.post("/api/v1/auth/login", json={"username": "stu_demo_01", "password": "Passw0rd123"})
    print("login:", r.status_code)
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("me:", client.get("/api/v1/auth/me", headers=headers).json())

    r = client.post("/api/v1/sessions", json={"title": "冒烟"}, headers=headers)
    sid = r.json()["id"]
    print("session:", r.status_code, sid)

    r = client.post(f"/api/v1/sessions/{sid}/messages",
                    json={"prompt": "01背包内层循环为什么必须倒序", "stream": False},
                    headers=headers, timeout=180)
    data = r.json()
    print("answer:", r.status_code, "| level:", data.get("level"), "| sources:",
          [s["title"] for s in data.get("sources", [])])
    print("answer preview:", (data.get("answer") or "")[:160].replace("\n", " "))

    print("stats:", client.get("/api/v1/stats/overview", headers=headers).json())
    print("readyz:", client.get("/readyz").status_code, client.get("/readyz").json())
    print("knowledge:", client.get("/api/v1/knowledge?size=3", headers=headers).json()["total"])
