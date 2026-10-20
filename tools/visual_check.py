"""看板的**视觉**冒烟检查：真的用 Chrome 把每一页渲染出来，量几何、留截图。

为什么需要它：`tools/check_dashboard.py` 走的是 AppTest，只跑 Python 侧——
它证明得了「页面不报错、取得到底层数据」，证明不了「画出来了」。已经出过的三类
问题它一个都查不到：七张图渲染成空盒子（Altair 的 selection 挂在了不进返回值的
图层上）、分面图撑破容器盖住右边一栏（顶层 width 对分面 spec 是每格宽度），
以及页面长到四五屏。这三类只有在浏览器里量才看得见。

零依赖：只用标准库 + 本机 Chrome，不装 playwright、不联网。

为什么不用 headless 的 `--screenshot --virtual-time-budget`：虚拟时钟会跑到
websocket 前面，只能拍到 Streamlit 的加载骨架。这里自己驱动 CDP，按真实时间等
「运行中」指示器消失、元素数量稳定，然后再量、再截。

用法（先起好看板：`.venv/bin/streamlit run dashboard/app.py`）：

    python3 tools/visual_check.py /tmp/shots            # 走侧栏里的全部页面
    python3 tools/visual_check.py /tmp/shots --no-shots # 只量几何，不存图
    python3 tools/visual_check.py /tmp/shots --full     # 截整页（默认只截首屏）

不给 url 就打开首页，再从侧栏导航里把每一页都走一遍。发现可见却塌陷的图、
比容器宽的图、没关掉的 Streamlit 外壳、页面报错、或页高超护栏（`MAX_SCREENS`），
退出码为 1。

**它只量默认页签**：`st.tabs` 里未激活的内容量出来是 0×0（属正常），所以带页签的
页面要另外点开复核——第 6 页那三个页签就是这么量过的。
"""
from __future__ import annotations

import base64
import json
import os
import socket
import struct
import subprocess
import sys
import time
import urllib.request

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9222
PROFILE = "/tmp/chrome-cdp-profile"
VIEW_W, VIEW_H = 1600, 1000          # 视口宽 / 首屏高
MAX_PAGE_H = 16000                   # 全页截图的高度上限，超了 Chrome 会截不动

# 页高护栏（屏数）。看板已经改成单页长文档，一根长页的总屏数没有参考价值
# ——整页 20 屏是形态本身决定的，压不下去也不该压。真正要拦的是「哪一段太长」：
# 某一段塞了十几张图、读者滚一屏半还没到下一段，那就该拆卡而不是继续堆。
# 所以下面这个数按主题块量，逐块判；整页只留一个「渲染没塌」的下限。
MAX_SECTION_SCREENS = 6.0
MIN_PAGE_SCREENS = 1.0


# ---------------------------------------------------------------- HTTP 侧
def http_json(path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


def http_put(path: str):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="PUT")
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())


# ------------------------------------------------------------- WebSocket
class WS:
    """刚好够用的 WebSocket 客户端：文本帧、掩码、分片、控制帧。"""

    def __init__(self, url: str):
        rest = url[len("ws://"):]
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port)), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(
            f"GET /{path} HTTP/1.1\r\nHost: {hostport}\r\n"
            f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("握手时连接被关闭")
            buf += chunk
        head, _, self.buf = buf.partition(b"\r\n\r\n")
        if b" 101 " not in head.split(b"\r\n")[0]:
            raise RuntimeError("WebSocket 握手失败：" + head.decode(errors="replace"))
        self._id = 0

    # -- 收 --------------------------------------------------------------
    def _exact(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("连接断了")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _read_frame(self):
        b0, b1 = self._exact(2)
        fin, opcode = b0 & 0x80, b0 & 0x0F
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._exact(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._exact(8))[0]
        mask = self._exact(4) if b1 & 0x80 else None
        payload = self._exact(n)
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return fin, opcode, payload

    # -- 发 --------------------------------------------------------------
    def _frame(self, payload: bytes, opcode: int = 0x1):
        n = len(payload)
        head = bytearray([0x80 | opcode])
        if n < 126:
            head.append(0x80 | n)
        elif n < 65536:
            head.append(0x80 | 126)
            head += struct.pack(">H", n)
        else:
            head.append(0x80 | 127)
            head += struct.pack(">Q", n)
        mask = os.urandom(4)
        head += mask
        self.sock.sendall(bytes(head) + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    # -- 协议 ------------------------------------------------------------
    def call(self, method: str, **params):
        self._id += 1
        mid = self._id
        self._frame(json.dumps({"id": mid, "method": method, "params": params}).encode())
        while True:
            fin, opcode, payload = self._read_frame()
            if opcode == 0x8:
                raise ConnectionError("服务器主动关闭")
            if opcode == 0x9:                       # ping -> pong
                self._frame(payload, opcode=0xA)
                continue
            if opcode == 0xA:                       # pong
                continue
            # 分片：攒到 FIN 为止
            while not fin:
                fin, op, more = self._read_frame()
                if op in (0x9, 0xA):
                    continue
                payload += more
            msg = json.loads(payload)
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method} 失败：{msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.call("Runtime.evaluate", expression=expr, returnByValue=True,
                      awaitPromise=True)
        return r.get("result", {}).get("value")


# ------------------------------------------------------------------ 主流程
def launch_chrome():
    subprocess.run(["pkill", "-f", f"remote-debugging-port={PORT}"],
                   capture_output=True)
    time.sleep(0.5)
    proc = subprocess.Popen(
        [CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
         "--no-default-browser-check", "--disable-extensions",
         f"--remote-debugging-port={PORT}", f"--user-data-dir={PROFILE}",
         f"--window-size={VIEW_W},{VIEW_H}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            http_json("/json/version")
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Chrome 没能在 60 秒内起来")


READY_JS = """(() => {
  const running = document.querySelector('[data-testid="stStatusWidget"]');
  const app = document.querySelector('[data-testid="stAppViewContainer"]');
  const els = document.querySelectorAll('[data-testid="stElementContainer"]').length;
  const charts = document.querySelectorAll('.vega-embed, canvas').length;
  const skel = document.querySelectorAll('.stSkeleton, [class*="skeleton"]').length;
  return JSON.stringify({running: !!running, app: !!app, els, charts, skel});
})()"""


def wait_ready(ws, quiet=3.0, timeout=120.0):
    """等到 Streamlit 的「运行中」消失、元素数量连续几秒不变。"""
    t0 = time.time()
    last, stable_since = None, time.time()
    while time.time() - t0 < timeout:
        raw = ws.js(READY_JS)
        st = json.loads(raw) if raw else {}
        sig = (st.get("els"), st.get("charts"), st.get("running"))
        if sig != last:
            last, stable_since = sig, time.time()
        if (st.get("app") and not st.get("running") and st.get("els", 0) > 3
                and time.time() - stable_since >= quiet):
            time.sleep(1.0)                      # 给图形库最后一点时间
            return True, st
        time.sleep(0.5)
    return False, last




def shoot(ws, out: str, full: bool = False) -> int:
    """截图。full=True 时先把视口拉到内容全高再截。

    不能用 captureBeyondViewport：Streamlit 滚的是内层 stMain，文档本身只有一屏高，
    getLayoutMetrics 量到的 cssContentSize 就是视口大小，截出来还是一屏。
    把视口高度直接设成 stMain.scrollHeight，才真能把整页拉直。
    """
    if full:
        raw = ws.js("(() => { const m = document.querySelector"
                    "('[data-testid=\"stMain\"]'); return m ? m.scrollHeight : 0; })()")
        h = int(raw or 0)
        ws.call("Emulation.setDeviceMetricsOverride", width=VIEW_W,
                height=min(max(h + 32, VIEW_H), MAX_PAGE_H),
                deviceScaleFactor=1, mobile=False)
        time.sleep(1.5)
    else:
        ws.call("Emulation.setDeviceMetricsOverride", width=VIEW_W, height=VIEW_H,
                deviceScaleFactor=1, mobile=False)
        time.sleep(0.8)
    r = ws.call("Page.captureScreenshot", format="png")
    with open(out, "wb") as f:
        f.write(base64.b64decode(r["data"]))
    return os.path.getsize(out)


AUDIT_JS = r"""(() => {
  const box = e => { const r = e.getBoundingClientRect();
                     return [Math.round(r.width), Math.round(r.height)]; };
  const main = document.querySelector('[data-testid="stMain"]');

  // 可见却塌陷的图 = 画崩了。藏在未激活的 tab / 收起的抽屉里的图量出来是 0×0，
  // 那是正常的，用 offsetParent 把两种情况分开。
  const charts = [...document.querySelectorAll('.vega-embed')].map((e, i) => {
    const [w, h] = box(e);
    const s = e.querySelector('svg');
    const sw = s ? Math.round(s.getBoundingClientRect().width) : 0;
    let hiddenBy = null;
    if (e.offsetParent === null) {
      hiddenBy = e.closest('[data-testid="stTabPanel"]') ? '未激活的 tab'
               : e.closest('[data-testid="stExpanderDetails"]') ? '收起的抽屉'
               : '其它隐藏祖先';
    }
    return {i, w, h, hiddenBy, svg: e.querySelectorAll('svg').length,
            sw, over: sw - w};
  });

  // 只查「读者不该看到的东西」。stHeader 是个一直存在的结构容器（toolbarMode
  // 设为 minimal 后里面是空的），量它有高度不算问题；Deploy 按钮、开发者菜单、
  // 运行指示器才是真信号。
  const chrome = ['stAppDeployButton', 'stMainMenu', 'stToolbar', 'stStatusWidget']
    .filter(id => { const e = document.querySelector(`[data-testid="${id}"]`);
                    return e && e.getBoundingClientRect().height > 0; });

  const ex = document.querySelector('[data-testid="stException"]');
  // 单页长文档：六个主题块靠 st.title 分界，量出每块的起止位置，
  // 页高护栏就从「整页几屏」改成「哪一段几屏」。
  const heads = [...document.querySelectorAll('[data-testid="stHeading"] h1')]
    .map(e => ({t: (e.innerText || '').trim().slice(0, 24),
                top: Math.round(e.getBoundingClientRect().top +
                                window.scrollY)}))
    .filter(h => h.top > 0);
  return JSON.stringify({
    h: main ? main.scrollHeight : 0,
    charts,
    heads,
    metrics: [...document.querySelectorAll('[data-testid="stMetric"]')].map(box),
    chrome,
    err: ex ? ex.innerText.slice(0, 300) : null,
  });
})()"""


def audit_page(ws, url: str, outdir: str, shots: bool, full: bool = False) -> bool:
    """量一页的几何、留一张截图，返回这一页是否通过。"""
    ws.call("Emulation.setDeviceMetricsOverride", width=VIEW_W, height=VIEW_H,
            deviceScaleFactor=1, mobile=False)
    ws.call("Page.navigate", url=url)
    ok, _ = wait_ready(ws, timeout=90)
    raw = ws.js(AUDIT_JS)
    d = json.loads(raw) if raw else {}
    slug = url.rstrip("/").split("/")[-1] or "index"

    h = d.get("h", 0)
    screens = h / VIEW_H
    charts = d.get("charts", [])
    shown = [c for c in charts if c["hiddenBy"] is None]
    broken = [c for c in shown if c["h"] < 30]
    # 分面图没定宽：顶层 width 对分面 spec 是「每格」的宽度，Streamlit 把它撑成
    # 容器宽，于是 k 格的图画出 k 倍宽，从卡片里溢出去压住右边那一栏。
    spill = [c for c in shown if c.get("over", 0) > 8]
    hidden = len(charts) - len(shown)

    problems = []
    if not ok:
        problems.append("等待超时（页面可能没渲染完）")
    if d.get("err"):
        problems.append(f"页面报错：{d['err'][:160]}")
    if broken:
        problems.append(f"{len(broken)} 张图可见却是空盒子（高度 {[c['h'] for c in broken]}）")
    if spill:
        problems.append(f"{len(spill)} 张图比容器宽，溢出 {[c['over'] for c in spill]}px"
                        "（分面图要用 charts.facet_width 定宽并关掉 use_container_width）")
    if d.get("chrome"):
        problems.append(f"Streamlit 外壳还露着：{d['chrome']}")

    # 护栏：按主题块量，而不是整页。st.title 的纵向位置是块与块的分界。
    heads = sorted(d.get("heads", []), key=lambda x: x["top"])
    bounds = heads + [{"t": "页尾", "top": h}]
    fat = []
    for a, b in zip(bounds, bounds[1:]):
        sec = (b["top"] - a["top"]) / VIEW_H
        if sec > MAX_SECTION_SCREENS:
            fat.append(f"{a['t']} {sec:.1f} 屏")
    if screens < MIN_PAGE_SCREENS:
        problems.append(f"页面只有 {screens:.1f} 屏，多半是没渲染出来")
    if fat:
        problems.append(f"这些段超过 {MAX_SECTION_SCREENS:.1f} 屏的护栏："
                        + "、".join(fat))

    metrics = d.get("metrics", [])
    print(f"[{'OK  ' if not problems else 'FAIL'}] {slug:26s} "
          f"整页 {screens:4.1f} 屏（{h}px）  "
          + ("屏 ".join(f"{a['t']} {((b['top'] - a['top']) / VIEW_H):.1f}"
                        for a, b in zip(bounds, bounds[1:])) + "  " if heads else "")
          + f"图 {len(shown)} 张"
          + (f" +{hidden} 张藏在 tab/抽屉里" if hidden else "")
          + (f"  读数卡 {len(metrics)} 张" if metrics else ""))
    if shots:
        path = os.path.join(outdir, f"{slug}.png")
        size = shoot(ws, path, full=full)
        print(f"       截图 {path}（{size // 1024} KB）")
    for p in problems:
        print(f"       ⚠ {p}")
    return not problems


def main() -> int:
    flags = {"--no-shots", "--full"}
    argv = [a for a in sys.argv[1:] if a not in flags]
    shots = "--no-shots" not in sys.argv
    full = "--full" in sys.argv
    outdir = argv[0] if argv else "/tmp/shots"
    urls = argv[1:]
    os.makedirs(outdir, exist_ok=True)

    proc = launch_chrome()
    bad = 0
    try:
        target = http_put("/json/new?about:blank")
        ws = WS(target["webSocketDebuggerUrl"])
        ws.call("Page.enable")
        ws.call("Runtime.enable")

        if not urls:
            # 看板已经是一张单页长文档：st.navigation 与侧栏目录都下线了，
            # 只有一个 URL 可审。传 URL 参数进来时仍可单独审某个地址。
            urls = ["http://localhost:8501"]

        for url in urls:
            if not audit_page(ws, url, outdir, shots, full=full):
                bad += 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    print()
    if bad:
        print(f"{bad} / {len(urls)} 页未通过。")
        return 1
    print(f"{len(urls)} 页全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
