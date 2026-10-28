"""桌面入口：python LearningLab.py。训练线程通过队列与 Tk 主线程通信。"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from Experiment import DEFAULT_CONFIG, LIVE_KEYS, TrainingSession, rolling_means, validate_config

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class LearningLab:
    def __init__(self, root, output_root=None):
        self.root = root
        self.root.title("GridWorld · Q-learning 学习实验室")
        self.root.geometry("1440x940")
        self.root.minsize(1120, 800)
        self.root.configure(bg="#eef2f7")
        if getattr(sys, "frozen", False):
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).resolve().parent
        self.output_root = Path(output_root or app_dir / "runs")
        self.events = queue.Queue()
        self.session = None
        self.worker = None
        self.training = []
        self.evaluations = []
        self.parameter_changes = []
        self.marker_lines = []
        self.structural_widgets = []
        self.variables = {}
        self.dirty = False
        self.last_draw = 0
        self.closing = False
        self.last_path = None
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self.poll)

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#eef2f7")
        style.configure("TLabel", background="#eef2f7", font=("Microsoft YaHei UI", 10))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=6)
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Note.TLabel", foreground="#526079", font=("Microsoft YaHei UI", 9))
        style.configure("TLabelframe", background="#eef2f7")
        style.configure("TLabelframe.Label", background="#eef2f7", font=("Microsoft YaHei UI", 10, "bold"))

        header = ttk.Frame(self.root, padding=(20, 14))
        header.pack(fill="x")
        ttk.Label(header, text="GridWorld · 学习实验室", style="Title.TLabel").pack(side="left")
        self.status = tk.StringVar(value="就绪 · 设置参数后开始训练")
        ttk.Label(header, textvariable=self.status).pack(side="right")

        self.split = ttk.Panedwindow(self.root, orient="vertical")
        self.split.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        upper = ttk.Frame(self.split)
        bottom = ttk.Frame(self.split)
        self.split.add(upper, weight=4)
        self.split.add(bottom, weight=1)
        self.root.after(200, self.position_sash)

        sidebar_container = ttk.Frame(upper, width=340, padding=(10, 0, 0, 0))
        sidebar_container.pack(side="right", fill="y")
        sidebar_container.pack_propagate(False)
        sidebar_canvas = tk.Canvas(sidebar_container, bg="#eef2f7", highlightthickness=0)
        sidebar_scroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_scroll.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar = ttk.Frame(sidebar_canvas)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def resize_sidebar(event):
            sidebar_canvas.itemconfigure(sidebar_window, width=event.width)

        def update_sidebar_region(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        sidebar_canvas.bind("<Configure>", resize_sidebar)
        sidebar.bind("<Configure>", update_sidebar_region)
        plot_area = ttk.Frame(upper)
        plot_area.pack(side="left", fill="both", expand=True)

        ttk.Label(
            plot_area,
            text="蓝色：训练滑动平均    橙色：冻结 Q 表、epsilon=0 的定期评估",
            style="Note.TLabel",
        ).pack(anchor="w", pady=(0, 5))
        self.figure = Figure(figsize=(9, 6), dpi=100, facecolor="#ffffff")
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self.figure.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.09, hspace=0.22)
        self.train_lines = []
        self.eval_lines = []
        titles = ["成功率 (%)", "平均单局总奖励", "平均单局动作数"]
        for index in range(3):
            axis = self.axes[index]
            axis.set_ylabel(titles[index], fontsize=10)
            axis.grid(True, color="#e5eaf1", linewidth=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            train_line, = axis.plot([], [], color="#2878dc", linewidth=1.7)
            eval_line, = axis.plot([], [], color="#ed8b26", linewidth=1.8, marker="o", markersize=3)
            self.train_lines.append(train_line)
            self.eval_lines.append(eval_line)
        self.axes[0].set_ylim(-3, 103)
        self.axes[-1].set_xlabel("已完成的训练局数")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_area, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.canvas.draw()

        form = ttk.LabelFrame(sidebar, text="实验参数", padding=10)
        form.pack(fill="x")
        fields = [
            ("mode", "地图模式", "fixed"),
            ("map_seed", "地图种子", 99999999),
            ("agent_seed", "智能体种子", 0),
            ("episodes", "训练局数", 1000),
            ("max_steps", "每局步数上限", 300),
            ("alpha", "学习率 α", 0.1),
            ("gamma", "折扣因子 γ", 0.95),
            ("epsilon", "探索概率 ε", 0.2),
            ("eval_interval", "每隔多少局评估", 50),
            ("eval_episodes", "每次评估局数", 20),
            ("window", "训练平滑窗口", 20),
        ]
        for row, field in enumerate(fields):
            key, label, default = field
            variable = tk.StringVar(value=str(default))
            self.variables[key] = variable
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            if key == "mode":
                widget = ttk.Combobox(form, textvariable=variable, values=["fixed", "random"],
                                      state="readonly", width=13)
            else:
                widget = ttk.Entry(form, textvariable=variable, width=16)
            widget.grid(row=row, column=1, sticky="ew", padx=(10, 0), pady=3)
            if key not in LIVE_KEYS and key != "window":
                self.structural_widgets.append((key, widget))
        form.columnconfigure(1, weight=1)

        controls = ttk.Frame(sidebar)
        controls.pack(fill="x", pady=10)
        self.start_button = ttk.Button(controls, text="开始 / 重新训练", command=self.start)
        self.start_button.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.pause_button = ttk.Button(controls, text="暂停", command=self.toggle_pause, state="disabled")
        self.pause_button.grid(row=1, column=0, sticky="ew", pady=(5, 0), padx=(0, 3))
        self.stop_button = ttk.Button(controls, text="停止", command=self.stop, state="disabled")
        self.stop_button.grid(row=1, column=1, sticky="ew", pady=(5, 0))
        ttk.Button(controls, text="应用 α / γ / ε（下一局）", command=self.apply_parameters).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(controls, text="更新平滑窗口", command=self.update_window).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(5, 0)
        )
        ttk.Button(controls, text="打开数据目录", command=self.open_output).grid(
            row=4, column=0, sticky="ew", pady=(5, 0), padx=(0, 3)
        )
        ttk.Button(controls, text="保存曲线 PNG", command=self.save_plot).grid(
            row=4, column=1, sticky="ew", pady=(5, 0)
        )
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.progress = ttk.Progressbar(sidebar, maximum=100)
        self.progress.pack(fill="x", pady=(0, 8))
        self.metrics = tk.StringVar(value="尚未开始评估")
        ttk.Label(sidebar, textvariable=self.metrics, wraplength=300).pack(anchor="w")
        self.stability = tk.StringVar(value="等待至少 5 个评估点")
        ttk.Label(sidebar, textvariable=self.stability, wraplength=300, foreground="#245b46").pack(
            anchor="w", pady=(10, 4)
        )
        ttk.Label(
            sidebar, style="Note.TLabel", wraplength=300,
            text="稳定判据：近 5 个同参数评估点成功率均≥95%，平均奖励和平均步数极差各≤2。仅为表现稳定提示，不证明最优或收敛。\n\n步数包含超时局。random 模式评估新地图；固定图重复成功不代表跨地图泛化。"
        ).pack(anchor="w", pady=(5, 0))

        log_header = ttk.Frame(bottom)
        log_header.pack(fill="x", pady=(6, 4))
        ttk.Label(log_header, text="运行日志 · 完整记录自动保存为 UTF-8 文件").pack(side="left")
        ttk.Button(log_header, text="清空显示", command=self.clear_log).pack(side="right")
        self.log_box = scrolledtext.ScrolledText(
            bottom, height=8, wrap="word", bg="#142035", fg="#d9e5f7",
            insertbackground="white", font=("Microsoft YaHei UI", 9), relief="flat"
        )
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

    def position_sash(self):
        try:
            height = self.split.winfo_height()
            self.split.sashpos(0, int(height * 0.77))
        except tk.TclError:
            pass

    def append_log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        # 界面仅保留最近 2000 行，完整日志已由工作线程保存。
        count = int(self.log_box.index("end-1c").split(".")[0])
        if count > 2000:
            self.log_box.delete("1.0", str(count - 2000) + ".0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def read_config(self):
        values = {}
        for key, variable in self.variables.items():
            values[key] = variable.get()
        return validate_config(values)

    def set_running(self, running):
        self.start_button.configure(state="disabled" if running else "normal")
        state = "normal" if running else "disabled"
        self.pause_button.configure(state=state, text="暂停")
        self.stop_button.configure(state=state)
        for key, widget in self.structural_widgets:
            if running:
                widget.configure(state="disabled")
            elif key == "mode":
                widget.configure(state="readonly")
            else:
                widget.configure(state="normal")

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            config = self.read_config()
        except (ValueError, OverflowError) as error:
            messagebox.showerror("参数不正确", str(error), parent=self.root)
            return
        self.training = []
        self.evaluations = []
        self.parameter_changes = []
        for marker in self.marker_lines:
            marker.remove()
        self.marker_lines = []
        self.clear_log()
        self.stability.set("等待至少 5 个评估点")
        self.metrics.set("训练中；评估曲线将在首个评估点出现")
        self.status.set("训练中")
        self.progress.configure(value=0, maximum=config["episodes"])
        self.dirty = True
        self.session = TrainingSession(config, self.emit, self.output_root)
        self.set_running(True)
        self.worker = threading.Thread(target=self.session.run, daemon=True)
        self.worker.start()

    def emit(self, kind, value):
        self.events.put((kind, value))

    def toggle_pause(self):
        if self.session is None:
            return
        if self.session.pause_event.is_set():
            self.session.pause_event.clear()
            self.pause_button.configure(text="暂停")
            self.status.set("训练 / 评估中")
        else:
            self.session.pause_event.set()
            self.pause_button.configure(text="继续")
            self.status.set("已请求暂停")

    def stop(self):
        if self.session is not None:
            self.session.stop_event.set()
            self.session.pause_event.clear()
            self.status.set("正在停止并保存数据")

    def apply_parameters(self):
        if self.worker is None or not self.worker.is_alive():
            self.append_log("参数已在右侧设置；点击“开始 / 重新训练”后生效。")
            return
        try:
            candidate = dict(self.session.initial_config)
            changes = {}
            for key in LIVE_KEYS:
                changes[key] = self.variables[key].get()
                candidate[key] = changes[key]
            validated = validate_config(candidate)
            for key in LIVE_KEYS:
                changes[key] = validated[key]
        except (ValueError, OverflowError) as error:
            messagebox.showerror("参数不正确", str(error), parent=self.root)
            return
        self.session.commands.put(changes)
        self.append_log("已提交参数修改，将在下一局训练开始时应用并记录。")

    def update_window(self):
        try:
            window = int(self.variables["window"].get())
            if window < 1:
                raise ValueError("窗口必须是正整数")
        except ValueError as error:
            messagebox.showerror("平滑窗口不正确", str(error), parent=self.root)
            return
        self.dirty = True

    def draw_curves(self):
        try:
            window = max(1, int(self.variables["window"].get()))
        except ValueError:
            window = 20
        x = []
        success = []
        rewards = []
        steps = []
        for point in self.training:
            x.append(point["episode"])
            success.append(point["success"] * 100)
            rewards.append(point["reward"])
            steps.append(point["steps"])
        series = [success, rewards, steps]
        eval_keys = ["success_rate", "reward", "steps"]
        for index in range(3):
            values = rolling_means(series[index], window)
            self.train_lines[index].set_data(x, values)
            eval_x = []
            eval_y = []
            for point in self.evaluations:
                eval_x.append(point["episode"])
                eval_y.append(point[eval_keys[index]])
            self.eval_lines[index].set_data(eval_x, eval_y)
            self.axes[index].relim()
            self.axes[index].autoscale_view()
            self.axes[index].set_xlim(0, max(10, len(x) + 1))
        self.axes[0].set_ylim(-3, 103)
        self.canvas.draw_idle()
        self.last_draw = time.monotonic()
        self.dirty = False

    def poll(self):
        processed = 0
        # 为每次消息处理设置时间预算，避免海量日志阻塞按钮和关闭操作。
        deadline = time.monotonic() + 0.015
        while processed < 100 and time.monotonic() < deadline:
            try:
                kind, value = self.events.get_nowait()
            except queue.Empty:
                break
            processed += 1
            if kind == "log":
                self.append_log(value)
            elif kind == "training":
                self.training.append(value)
                self.progress.configure(value=value["episode"])
                self.dirty = True
            elif kind == "evaluation":
                self.evaluations.append(value)
                self.metrics.set(
                    f"最近评估：第 {value['episode']} 局\n"
                    f"成功率 {value['success_rate']:.1f}% · 奖励 {value['reward']:.2f}\n"
                    f"步数 {value['steps']:.1f} · 碰撞 {value['collisions']:.1f}"
                )
                self.stability.set(value["stability"])
                self.dirty = True
            elif kind == "parameters":
                self.parameter_changes.append(value)
                for axis in self.axes:
                    marker = axis.axvline(value["episode"], color="#9f74b5", linestyle=":", alpha=0.7)
                    self.marker_lines.append(marker)
                self.stability.set("参数已变化，重新积累评估点")
            elif kind == "error":
                self.status.set("发生错误，请查看下方日志")
                self.append_log(value)
            elif kind == "finished":
                self.last_path = Path(value["path"])
                self.set_running(False)
                self.status.set(value["status"] + f" · 已完成 {value['completed']} 局")
                self.append_log("数据已保存：" + value["path"])
                self.dirty = True
        if self.dirty and time.monotonic() - self.last_draw > 0.3:
            self.draw_curves()
        if self.closing:
            if self.worker is None or not self.worker.is_alive():
                self.root.destroy()
                return
        self.root.after(80, self.poll)

    def open_output(self):
        path = self.last_path
        if path is None:
            path = self.output_root
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(path))
        except OSError as error:
            messagebox.showerror("无法打开目录", str(error), parent=self.root)

    def save_plot(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension=".png",
            filetypes=[("PNG 图片", "*.png")], initialfile="learning_curve.png"
        )
        if path:
            self.draw_curves()
            self.figure.savefig(path, dpi=180, facecolor="white")
            self.append_log("曲线已保存：" + path)

    def close(self):
        self.closing = True
        self.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", help="自动运行小实验并保存验证产物")
    args = parser.parse_args()
    root = tk.Tk()
    app = LearningLab(root, args.smoke_test)
    if args.smoke_test:
        target = Path(args.smoke_test)
        target.mkdir(parents=True, exist_ok=True)
        app.variables["episodes"].set("30")
        app.variables["eval_interval"].set("5")
        app.variables["eval_episodes"].set("2")
        root.after(400, app.start)
        deadline = time.monotonic() + 90

        def finish_smoke():
            if time.monotonic() > deadline:
                app.stop()
                root.destroy()
                raise RuntimeError("GUI smoke test timed out")
            if app.worker is not None and not app.worker.is_alive() and len(app.training) == 30:
                app.draw_curves()
                app.canvas.draw()
                app.figure.savefig(target / "curves.png", dpi=120)
                root.update_idletasks()
                report = {"training_points": len(app.training), "evaluation_points": len(app.evaluations)}
                (target / "gui_report.json").write_text(json.dumps(report), encoding="utf-8")
                root.destroy()
                return
            root.after(100, finish_smoke)

        root.after(600, finish_smoke)
    root.mainloop()


if __name__ == "__main__":
    main()
