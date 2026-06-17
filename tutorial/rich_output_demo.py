"""Demonstrate common Rich terminal output renderables.

Run:

    /home/leo/miniconda3/bin/conda run -n sfquant python tutorial/rich_output_demo.py
"""

from __future__ import annotations

import json
import time

try:
    from rich import inspect as rich_inspect
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        track,
    )
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.traceback import install as install_rich_traceback
    from rich.tree import Tree
except ImportError as exc:
    raise SystemExit("缺少 rich，请先安装：pip install rich") from exc


def main() -> None:
    console = Console()

    console.print(Rule("[bold cyan]Rich 常用输出类型演示"))
    print_supported_renderables(console)
    print_markdown(console)
    print_table(console)
    print_panel(console)
    print_syntax(console)
    print_json(console)
    print_tree(console)
    print_columns(console)
    print_progress(console)
    print_live_training_metrics(console)
    print_layout(console)
    print_status(console)
    print_spinner(console)
    print_advanced_progress(console)
    print_inspect(console)
    print_traceback_install(console)
    print_exception(console)
    console.print(Rule("[bold green]演示结束"))


def print_supported_renderables(console: Console) -> None:
    table = Table(title="本 demo 覆盖的常用输出")
    table.add_column("类型", style="cyan", no_wrap=True)
    table.add_column("用途")
    table.add_row("Text Style", "彩色、加粗、下划线等终端文本样式")
    table.add_row("Markdown", "把 Markdown 标题、列表、代码块渲染到终端")
    table.add_row("Table", "配置、指标、实验结果等表格")
    table.add_row("Panel", "带边框的信息块")
    table.add_row("Syntax", "代码语法高亮")
    table.add_row("JSON / Pretty", "格式化输出 dict、list、JSON")
    table.add_row("Tree", "目录树、层级结构")
    table.add_row("Columns", "多列展示短项")
    table.add_row("Rule", "分隔线和章节标题")
    table.add_row("Progress", "进度条")
    table.add_row("Live", "动态刷新一块区域，比如实时训练指标表")
    table.add_row("Layout", "终端分屏布局")
    table.add_row("Status", "带状态文字的加载中区域")
    table.add_row("Spinner", "转圈等待动画")
    table.add_row("inspect()", "漂亮地查看 Python 对象")
    table.add_row("traceback.install()", "全局替换未捕获异常的显示样式")
    table.add_row("Traceback", "更易读的异常堆栈")
    console.print(table)


def print_markdown(console: Console) -> None:
    console.print(Rule("[bold]Markdown"))
    markdown = """
# 3DGS 训练摘要

- 当前模式：`standard_3dgs`
- 输出目录：`outputs/3dgs_scene`
- 关键指标：**PSNR**、**SSIM**、**LPIPS**

```python
loss, parts = photometric_loss(render.image, gt_image)
```

> Markdown 在这里是渲染到终端，不是写出 `.md` 文件。
"""
    console.print(Markdown(markdown))


def print_table(console: Console) -> None:
    console.print(Rule("[bold]Table"))
    table = Table(title="3DGS 训练配置")
    table.add_column("项目", style="cyan", no_wrap=True)
    table.add_column("值")
    table.add_row("设备", "CUDA GPU='NVIDIA GeForce RTX 3070 Laptop GPU'")
    table.add_row("分辨率", "1600x1065")
    table.add_row("训练步数", "19999")
    table.add_row("densification_mode", "standard_3dgs")
    console.print(table)


def print_panel(console: Console) -> None:
    console.print(Rule("[bold]Panel"))
    console.print(
        Panel(
            "训练前检查：CUDA 可用，COLMAP 点云存在，输出目录已创建。",
            title="启动检查",
            border_style="green",
        )
    )


def print_syntax(console: Console) -> None:
    console.print(Rule("[bold]Syntax"))
    code = """\
def resolve_densify_stop_step(iterations: int, requested_stop_step: int) -> int:
    if requested_stop_step > 0:
        return int(requested_stop_step)
    return max(min(int(iterations) - 500, 15_000), 501)
"""
    console.print(Syntax(code, "python", theme="monokai", line_numbers=True))


def print_json(console: Console) -> None:
    console.print(Rule("[bold]JSON / Pretty"))
    payload = {
        "metrics": {"psnr": 26.42, "ssim": 0.8123, "lpips": 0.1841},
        "best_checkpoint": "outputs/3dgs_scene/best/best_psnr.ply",
        "enabled": ["eval", "densification"],
    }
    console.print_json(json.dumps(payload, ensure_ascii=False))
    console.print({"raw_dict_pretty": payload})


def print_tree(console: Console) -> None:
    console.print(Rule("[bold]Tree"))
    tree = Tree("outputs/3dgs_scene")
    tree.add("final.ply")
    tree.add("training_summary.json")
    checkpoints = tree.add("checkpoints")
    checkpoints.add("step_001000.ply")
    checkpoints.add("step_002000.ply")
    previews = tree.add("previews")
    previews.add("step_001000.png")
    previews.add("step_002000.png")
    console.print(tree)


def print_columns(console: Console) -> None:
    console.print(Rule("[bold]Columns"))
    modes = [
        Panel("standard_3dgs", style="cyan"),
        Panel("patch_guided", style="magenta"),
        Panel("patch_reallocate", style="yellow"),
        Panel("official_3dgs LPIPS", style="green"),
    ]
    console.print(Columns(modes, equal=True, expand=True))


def print_progress(console: Console) -> None:
    console.print(Rule("[bold]Progress"))
    for _ in track(range(8), description="模拟评估 test views"):
        time.sleep(0.03)


def print_live_training_metrics(console: Console) -> None:
    console.print(Rule("[bold]Live 实时刷新训练指标表"))
    with Live(_training_metrics_table(0, 0.0, 0.0, 25_837), console=console, refresh_per_second=8) as live:
        for step in range(0, 101, 10):
            loss = 0.42 / (1.0 + step * 0.04)
            psnr = 18.0 + step * 0.08
            gaussians = 25_837 + step * 32
            live.update(_training_metrics_table(step, loss, psnr, gaussians))
            time.sleep(0.08)


def _training_metrics_table(step: int, loss: float, psnr: float, gaussians: int) -> Table:
    table = Table(title="实时训练指标")
    table.add_column("step", justify="right")
    table.add_column("loss", justify="right")
    table.add_column("PSNR", justify="right")
    table.add_column("Gaussian", justify="right")
    table.add_row(str(step), f"{loss:.4f}", f"{psnr:.2f}", str(gaussians))
    return table


def print_layout(console: Console) -> None:
    console.print(Rule("[bold]Layout 终端分屏"))
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(name="body"))
    layout["body"].split_row(Layout(name="left"), Layout(name="right"))
    layout["header"].update(Panel("3DGS Dashboard", style="bold cyan"))
    layout["left"].update(Panel("训练状态\nstep=1000\nloss=0.0312", title="Train"))
    layout["right"].update(Panel("评估指标\nPSNR=25.81\nSSIM=0.8123", title="Eval"))
    console.print(layout)


def print_status(console: Console) -> None:
    console.print(Rule("[bold]Status"))
    with console.status("[bold green]正在加载 COLMAP 数据...", spinner="dots"):
        time.sleep(0.4)
    console.print("COLMAP 数据加载完成")


def print_spinner(console: Console) -> None:
    console.print(Rule("[bold]Spinner"))
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("等待 LPIPS 网络加载", total=None)
        time.sleep(0.5)
    console.print("LPIPS 网络加载完成")


def print_advanced_progress(console: Console) -> None:
    console.print(Rule("[bold]Progress 高级用法：多任务、ETA、百分比"))
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    with progress:
        train_task = progress.add_task("训练", total=100)
        eval_task = progress.add_task("评估", total=60)
        for index in range(20):
            progress.update(train_task, advance=5)
            if index % 2 == 0:
                progress.update(eval_task, advance=6)
            time.sleep(0.04)


def print_inspect(console: Console) -> None:
    console.print(Rule("[bold]inspect()"))
    config = {
        "data": "src/datasets/SeathruNeRF_dataset/Curasao",
        "iterations": 19_999,
        "densification": {"mode": "standard_3dgs", "start": 1000, "stop": 7500},
    }
    rich_inspect(config, title="训练配置对象", methods=False)


def print_traceback_install(console: Console) -> None:
    console.print(Rule("[bold]rich.traceback.install()"))
    install_rich_traceback(show_locals=False)
    console.print("已调用 rich.traceback.install(show_locals=False)。")
    console.print("之后如果有未捕获异常，Rich 会接管默认异常显示。")


def print_exception(console: Console) -> None:
    console.print(Rule("[bold]Traceback"))
    try:
        raise ValueError("示例异常：semantic_importance_root 为空")
    except ValueError:
        console.print_exception(show_locals=False)


if __name__ == "__main__":
    main()
