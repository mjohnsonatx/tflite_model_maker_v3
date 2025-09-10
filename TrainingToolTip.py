import tkinter as tk


class ToolTip:
    """Simple tooltip class"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.enabled = True
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        if self.enabled:
            self.schedule_show()

    def leave(self, event=None):
        self.cancel_show()
        self.hide()

    def schedule_show(self):
        self.cancel_show()
        self.show_timer = self.widget.after(500, self.show)

    def cancel_show(self):
        if hasattr(self, 'show_timer'):
            self.widget.after_cancel(self.show_timer)

    def show(self):
        if self.tipwindow or not self.enabled:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = (tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1))
        label.pack(ipadx=1)

    def hide(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()