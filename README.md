# GridWorld Q-learning 学习实验室

桌面界面：左侧学习曲线、右侧参数、下方运行日志。原来的 Main.py 终端训练入口仍可使用。

## 启动

优先双击 `dist/GridWorldLearningLab.exe`，无需另外安装 Python。

源码启动：

```powershell
python LearningLab.py
```

源码界面需要 Python 自带的 Tkinter 和 Matplotlib：

```powershell
python -m pip install -r requirements.txt
```

## 操作

1. 选择 fixed 或 random，设置地图种子和智能体种子。默认训练 1000 局，每局最多 300 步。
2. 点击“开始 / 重新训练”。训练在工作线程执行，可以暂停、继续、停止。
3. 运行时修改 α、γ、ε，再点击“应用”。参数在下一局训练开始时生效，曲线上以紫色虚线标记，完整变化保存在日志和 JSON 中。
4. 地图种子、智能体种子、训练局数等参数需要停止后重新开始。重新开始会创建新的 Q 表，不接续上一轮。
5. “更新平滑窗口”只调整训练曲线的显示，不改变训练或评估数据。
6. 右侧参数区域可用滚动条滚动。底部日志栏和上方图表区域之间可以拖动调整高度。
7. 工具栏可缩放、平移图表，“保存曲线 PNG”可以导出当前曲线。

## 曲线与评估

- 蓝色：每局训练数据的滑动平均。
- 橙色：每隔指定局数执行的独立贪心评估，epsilon 固定为 0，不调用 learn。
- 成功率：到达终点的回合比例。
- 奖励：单局累计奖励。
- 步数：单局动作总数，包含碰撞及达到步数上限的失败局，避免只统计成功局而美化结果。

训练和评估使用独立环境及独立动作随机数生成器，评估不会消耗训练的随机序列。改变评估频率不会改变同参数训练轨迹。

fixed 评估使用与训练相同地图。random 评估使用独立种子（地图种子 + 1000003）的新地图序列，衡量新地图表现；当前表格 Q-learning 不擅长跨地图泛化。epsilon=0 时，并列最优动作仍会随机选择。

稳定提示使用最近 5 个相同参数版本的评估点：成功率均至少 95%，平均奖励极差不超过 2，平均步数极差不超过 2。改变 α、γ、ε 后会重新积累稳定性证据。这个提示不是 Q 值数学收敛、最优策略或跨地图泛化的证明，也不自动停止训练。更严谨的比较应使用多个智能体种子分别重训。

## 数据

每次实验创建一个唯一的 runs 子目录：

- episodes.csv：逐回合训练和评估记录、奖励、步数、成功、碰撞、学习参数、Q 表大小。
- run.log：完整 UTF-8 中文日志。界面只保留最近 2000 行，文件保留全部。
- summary.json：起始/最终参数、参数变化、评估点、稳定状态和实验结束状态。

CSV 带 UTF-8 BOM，便于 Excel 打开。源码模式数据在脚本目录的 runs 中；EXE 模式在 EXE 同目录的 runs 中。请将 EXE 放在可写目录运行。“打开数据目录”可直接定位结果。

## 文件结构

- Environment.py：10×10 地图和规则，包含 BFS 生成检查。普通移动与碰撞 -1，奖励 +5，终点净 +10。
- MapChecker.py：地图连通性检查，奖励必须在进入终点前可达。
- QAgent.py：Q 表、动作选择、单步更新；可传入独立随机数生成器，旧调用方式仍兼容。
- Experiment.py：训练、定期评估、参数修改、数据持久化。
- LearningLab.py：Tkinter + Matplotlib 桌面界面。
- Main.py：原有终端入口。
- BuildExe.py：本机打包脚本。

## 测试与打包

```powershell
python -m unittest -v
python -m pip install pyinstaller
python BuildExe.py
```

BuildExe.py 优先使用项目根目录 .build_tools 中的打包工具，否则使用当前 Python 已安装的 PyInstaller。产物为 dist/GridWorldLearningLab.exe。

可运行一次界面自动检查：

```powershell
python LearningLab.py --smoke-test build/gui_smoke
```

它会自动完成 30 局训练及 6 次评估，保存曲线和验证报告后退出。

实现参考：[Matplotlib 嵌入 Tk 官方示例](https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html)、[Tkinter 线程模型](https://docs.python.org/3/library/tkinter.html#threading-model)、[PyInstaller 官方用法](https://pyinstaller.org/en/stable/usage.html)。
