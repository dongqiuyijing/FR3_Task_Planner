import tkinter as tk
import time
import threading
class LongPressButton:
    def __init__(self, parent, text=None, press_cmd=None, release_cmd=None, 
                 long_thresh=0.8):
        self.btn = tk.Button(parent, text=text)
        
        # 存储长按相关参数
        self._press_cmd = press_cmd  # 按下时执行的命令
        self._release_cmd = release_cmd  # 松开时执行的命令
        self._long_thresh = long_thresh  # 长按判定阈值
        
        # 事件绑定
        self.btn.bind("<ButtonPress-1>", self._on_press)
        self.btn.bind("<ButtonRelease-1>", self._on_release)
        self.btn.bind("<Leave>", self._on_leave)
        
        # 状态变量
        self._press_time = 0
        self._is_pressing = False
        self._long_triggered = False
        self._timer_id = None
    def _on_press(self, event):
        self._is_pressing = True
        self._press_time = time.time()
        
        # 设置定时器检查是否长按
        self._timer_id = self.btn.after(
            int(self._long_thresh * 1000),
            self._check_long_press
        )
        
        # 立即执行按下命令
        if self._press_cmd:
            self._press_cmd()
    def _check_long_press(self):
        """标记已触发长按（但不再重复执行）"""
        if self._is_pressing:
            self._long_triggered = True
    def _on_release(self, event):
        self._cancel_timer()
        
        # 执行松开命令
        if self._release_cmd:
            self._release_cmd()
            
        self._reset_state()
    def _on_leave(self, event):
        """鼠标离开按钮时取消操作"""
        self._cancel_timer()
        if self._is_pressing and self._release_cmd:
            self._release_cmd()  # 如果鼠标移出时仍处于按下状态，也发送停止指令
        self._reset_state()

    def _cancel_timer(self):
        if self._timer_id:
            self.btn.after_cancel(self._timer_id)
            self._timer_id = None

    def _reset_state(self):
        self._is_pressing = False
        self._long_triggered = False
        self._press_time = 0

    def config(self, **kwargs):
        """处理配置选项"""
        if 'press_cmd' in kwargs:
            self._press_cmd = kwargs.pop('press_cmd')
        if 'release_cmd' in kwargs:
            self._release_cmd = kwargs.pop('release_cmd')
        if 'long_thresh' in kwargs:
            self._long_thresh = kwargs.pop('long_thresh')
        
        # 剩余参数传给标准按钮
        if kwargs:
            self.btn.config(**kwargs)
    # 布局方法
    def pack(self, **kwargs):
        self.btn.pack(**kwargs)
    def place(self, **kwargs):
        self.btn.place(**kwargs)
    def grid(self, **kwargs):
        self.btn.grid(**kwargs)

class Button:
    def __init__(self, parent):
        self.btn = tk.Button(parent)
        self.parent = parent
        self._external_command = None  # 外部函数
        self.close_frame = False # 是否关闭如今界面
        self.open_frame = None
        self.button_state = False
        self.pressed_text = None
        self.press_text = None
    def config(self, **kwargs):
        """处理配置选项
        
        Args:
            **kwargs: 配置选项，支持以下特殊参数:
                - button_state: 按钮状态回调函数
                - pressed_text: 按钮按下时显示的文本
                - press_text: 按钮文字
                - open_frame: 打开框架回调函数
                - close_frame: 关闭框架回调函数
                - command: 按钮命令回调函数(会与内部处理逻辑结合)
                其他参数将直接传递给按钮的config方法
        """
        # 定义特殊参数处理映射 (参数名: (属性名, 后处理函数))
        SPECIAL_ARGS = {
            'button_state': ('button_state', None),
            'pressed_text': ('pressed_text', None),
            'press_text': ('press_text', None),
            'open_frame': ('open_frame', None),
            'close_frame': ('close_frame', None),
            'command': ('_external_command', lambda: self.btn.config(command=self._combined_command))
        }
        
        # 处理特殊参数
        for arg, (attr, post_process) in SPECIAL_ARGS.items():
            if arg in kwargs:
                setattr(self, attr, kwargs.pop(arg))
                if post_process:
                    post_process()
        
        # 剩余参数传递给按钮配置
        if kwargs:
            self.btn.config(**kwargs)
    
    def set_internal_command(self):
        """设置内部执行的函数"""
        self.btn.config(text=self.press_text if self.button_state else self.pressed_text)
        if self.close_frame:
            self.parent.grid_forget()
        if self.open_frame:
            self.open_frame.grid()
    
    def _combined_command(self):
        """组合执行的函数"""
        self.set_internal_command()
        if self._external_command:
            self._external_command()  # 再执行外部函数
    
    # 布局方法
    def pack(self, **kwargs):
        self.btn.pack(**kwargs)
    
    def place(self, **kwargs):
        self.btn.place(**kwargs)
    
    def grid(self, **kwargs):
        self.btn.grid(**kwargs)



class TaskController:# 支持接收信息开始，再次接收信息停止的任务控制器
    """支持自定义任务函数的任务控制器"""
    def __init__(self, task_function=None):
        self.is_running = False
        self.lock = threading.Lock()
        self.task_thread = None
        self.thread_ready = True
        self.task_function = task_function or self.default_task
    def default_task(self):
        """默认任务函数"""
        print("正在执行默认任务...")
        time.sleep(1)
    def task_loop(self):
        """任务主循环"""
        try:
            while True:
                with self.lock:
                    if not self.is_running:
                        break
                self.task_function()
        except Exception as e:
            print("任务执行出错: %s", str(e))
        finally:
            with self.lock:
                self.thread_ready = True
            print("任务线程已安全退出")
    def start_task(self):
        """启动任务"""
        with self.lock:
            if not self.is_running:
                self.is_running = True
                if self.thread_ready or not self.task_thread.is_alive():
                    self.thread_ready = False
                    self.task_thread = threading.Thread(target=self.task_loop)
                    self.task_thread.start()
                    print("任务线程已启动")
    def stop_task(self):
        """停止任务"""
        with self.lock:
            if self.is_running:
                self.is_running = False
                print("已发送停止信号")