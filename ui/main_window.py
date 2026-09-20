import tkinter as tk

from common.types import ForgottenEvent
from memory.engine import DeskMemoryEngine


class DeskMemoryUI:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title("DeskMemory")
        self.root.geometry("1000x680")
        self.root.minsize(960, 640)

        # 当前 UI 对应的 DeskMemoryEngine
        # 供重新校准按钮使用
        self.engine: DeskMemoryEngine | None = None

        # ==================================================
        # Theme
        # ==================================================

        self.bg = "#08111f"
        self.header_bg = "#0b1628"

        self.card = "#111c2e"
        self.card_hover = "#172438"
        self.border = "#243247"

        self.text_primary = "#f8fafc"
        self.text_secondary = "#94a3b8"
        self.text_dim = "#64748b"

        self.green = "#2dd4a7"
        self.yellow = "#fbbf24"
        self.red = "#ff5c6c"
        self.red_dark = "#34151d"
        self.blue = "#4cc9f0"

        self.root.configure(
            bg=self.bg
        )

        # ==================================================
        # Header
        # ==================================================

        header = tk.Frame(
            self.root,
            bg=self.header_bg,
            height=100,
        )

        header.pack(
            fill="x"
        )

        header.pack_propagate(False)

        # --------------------------------------------------
        # Brand
        # --------------------------------------------------

        brand_area = tk.Frame(
            header,
            bg=self.header_bg,
        )

        brand_area.pack(
            side="left",
            padx=36,
            pady=18,
        )

        tk.Label(
            brand_area,
            text="DeskMemory",
            font=(
                "Segoe UI",
                25,
                "bold"
            ),
            fg=self.text_primary,
            bg=self.header_bg,
        ).pack(
            anchor="w"
        )

        tk.Label(
            brand_area,
            text="Smart Desk Memory System",
            font=(
                "Segoe UI",
                10
            ),
            fg=self.text_secondary,
            bg=self.header_bg,
        ).pack(
            anchor="w",
            pady=(2, 0),
        )

        # --------------------------------------------------
        # System status + Recalibrate
        # --------------------------------------------------

        status_area = tk.Frame(
            header,
            bg=self.header_bg,
        )

        status_area.pack(
            side="right",
            padx=36,
            pady=13,
        )

        self.system_status = tk.Label(
            status_area,
            text="●  SYSTEM ONLINE",
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            fg=self.green,
            bg=self.header_bg,
        )

        self.system_status.pack(
            anchor="e"
        )

        self.recalibrate_button = tk.Button(
            status_area,
            text="RECALIBRATE DESK",
            font=(
                "Segoe UI",
                9,
                "bold"
            ),
            fg=self.text_primary,
            bg=self.card_hover,
            activeforeground=self.text_primary,
            activebackground=self.border,
            relief="flat",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2",
            command=self._recalibrate,
        )

        self.recalibrate_button.pack(
            anchor="e",
            pady=(8, 0),
        )

        # --------------------------------------------------
        # Accent line
        # --------------------------------------------------

        tk.Frame(
            self.root,
            bg=self.blue,
            height=2,
        ).pack(
            fill="x"
        )

        # ==================================================
        # Main content
        # ==================================================

        self.main = tk.Frame(
            self.root,
            bg=self.bg,
        )

        self.main.pack(
            fill="both",
            expand=True,
            padx=36,
            pady=28,
        )

        self.main.grid_columnconfigure(
            0,
            weight=1,
        )

        self.main.grid_columnconfigure(
            1,
            weight=1,
        )

        self.main.grid_rowconfigure(
            1,
            weight=1,
        )

        # ==================================================
        # User State Card
        # ==================================================

        state_card = self._create_card(
            self.main,
            "CURRENT USER STATE",
        )

        state_card.grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(0, 18),
        )

        state_inner = tk.Frame(
            state_card,
            bg=self.card,
        )

        state_inner.pack(
            fill="x",
            padx=24,
            pady=(3, 22),
        )

        self.state_dot = tk.Label(
            state_inner,
            text="●",
            font=(
                "Segoe UI",
                24,
                "bold"
            ),
            fg=self.text_secondary,
            bg=self.card,
        )

        self.state_dot.pack(
            side="left"
        )

        self.user_state_label = tk.Label(
            state_inner,
            text="EMPTY",
            font=(
                "Segoe UI",
                24,
                "bold"
            ),
            fg=self.text_secondary,
            bg=self.card,
        )

        self.user_state_label.pack(
            side="left",
            padx=(14, 0),
        )

        self.user_state_description = tk.Label(
            state_inner,
            text="Calibrating empty desk baseline",
            font=(
                "Segoe UI",
                10
            ),
            fg=self.text_secondary,
            bg=self.card,
        )

        self.user_state_description.pack(
            side="right"
        )

        # ==================================================
        # Baseline Card
        # ==================================================

        baseline_card = self._create_card(
            self.main,
            "BASELINE OBJECTS",
        )

        baseline_card.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=(0, 9),
        )

        self.baseline_container = tk.Frame(
            baseline_card,
            bg=self.card,
        )

        self.baseline_container.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 18),
        )

        # ==================================================
        # Session Card
        # ==================================================

        session_card = self._create_card(
            self.main,
            "SESSION OBJECTS",
        )

        session_card.grid(
            row=1,
            column=1,
            sticky="nsew",
            padx=(9, 0),
        )

        self.session_container = tk.Frame(
            session_card,
            bg=self.card,
        )

        self.session_container.pack(
            fill="both",
            expand=True,
            padx=18,
            pady=(0, 18),
        )

        # ==================================================
        # Forgotten Alert Overlay
        # ==================================================

        self.alert_overlay = tk.Frame(
            self.root,
            bg=self.red_dark,
            highlightthickness=3,
            highlightbackground=self.red,
        )

        self.alert_icon = tk.Label(
            self.alert_overlay,
            text="⚠",
            font=(
                "Segoe UI Symbol",
                44,
                "bold"
            ),
            fg=self.yellow,
            bg=self.red_dark,
        )

        self.alert_icon.pack(
            pady=(24, 4)
        )

        tk.Label(
            self.alert_overlay,
            text="POSSIBLE FORGOTTEN ITEM",
            font=(
                "Segoe UI",
                11,
                "bold"
            ),
            fg=self.red,
            bg=self.red_dark,
        ).pack()

        self.alert_item_label = tk.Label(
            self.alert_overlay,
            text="",
            font=(
                "Segoe UI",
                30,
                "bold"
            ),
            fg=self.text_primary,
            bg=self.red_dark,
            wraplength=470,
            justify="center",
        )

        self.alert_item_label.pack(
            padx=50,
            pady=(10, 8),
        )

        tk.Label(
            self.alert_overlay,
            text="Please check your belongings before leaving",
            font=(
                "Segoe UI",
                11
            ),
            fg="#fda4af",
            bg=self.red_dark,
        ).pack(
            pady=(0, 24)
        )

        # 默认不显示 alert_overlay

    # ==================================================
    # UI Helpers
    # ==================================================

    def _create_card(
        self,
        parent,
        title,
    ):

        card = tk.Frame(
            parent,
            bg=self.card,
            highlightthickness=1,
            highlightbackground=self.border,
        )

        header = tk.Frame(
            card,
            bg=self.card,
        )

        header.pack(
            fill="x",
            padx=20,
            pady=(18, 14),
        )

        tk.Label(
            header,
            text=title,
            font=(
                "Segoe UI",
                10,
                "bold"
            ),
            fg=self.text_secondary,
            bg=self.card,
        ).pack(
            side="left"
        )

        return card

    def _clear_container(
        self,
        container,
    ):

        for child in container.winfo_children():
            child.destroy()

    def _empty_row(
        self,
        container,
        text,
    ):

        row = tk.Frame(
            container,
            bg=self.card_hover,
        )

        row.pack(
            fill="x",
            pady=5,
        )

        tk.Label(
            row,
            text=text,
            font=(
                "Segoe UI",
                10
            ),
            fg=self.text_dim,
            bg=self.card_hover,
        ).pack(
            anchor="w",
            padx=14,
            pady=12,
        )

    def _add_object_row(
        self,
        container,
        name,
        status=None,
    ):

        row = tk.Frame(
            container,
            bg=self.card_hover,
            highlightthickness=1,
            highlightbackground=self.border,
        )

        row.pack(
            fill="x",
            pady=5,
        )

        # --------------------------------------------------
        # Status dot
        # --------------------------------------------------

        dot_color = self.blue

        if status == "PRESENT":
            dot_color = self.green

        elif status == "REMOVED":
            dot_color = self.text_dim

        tk.Label(
            row,
            text="●",
            font=(
                "Segoe UI",
                9
            ),
            fg=dot_color,
            bg=self.card_hover,
        ).pack(
            side="left",
            padx=(14, 8),
        )

        # --------------------------------------------------
        # Object name
        # --------------------------------------------------

        tk.Label(
            row,
            text=name,
            font=(
                "Segoe UI",
                11,
                "bold"
            ),
            fg=self.text_primary,
            bg=self.card_hover,
        ).pack(
            side="left",
            pady=12,
        )

        # --------------------------------------------------
        # Status
        # --------------------------------------------------

        if status is not None:

            if status == "PRESENT":

                badge_fg = self.green
                badge_text = "PRESENT"

            else:

                badge_fg = self.text_secondary
                badge_text = "REMOVED"

            tk.Label(
                row,
                text=badge_text,
                font=(
                    "Segoe UI",
                    9,
                    "bold"
                ),
                fg=badge_fg,
                bg=self.card_hover,
            ).pack(
                side="right",
                padx=14,
            )

    # ==================================================
    # Alert
    # ==================================================

    def show_alert(
        self,
        names,
    ):

        display_name = "\n".join(
            name.replace(
                "_",
                " "
            ).upper()
            for name in names
        )

        self.alert_item_label.config(
            text=display_name
        )

        self.alert_overlay.place(
            relx=0.5,
            rely=0.54,
            anchor="center",
            width=580,
            height=300,
        )

        self.alert_overlay.lift()

    def hide_alert(self):

        self.alert_overlay.place_forget()

    # ==================================================
    # Recalibrate Desk
    # ==================================================

    def _recalibrate(self):

        # 还没有收到 Engine 时不进行任何操作
        if self.engine is None:
            return

        # ----------------------------------------------
        # 重置核心 DeskMemory 系统
        # ----------------------------------------------

        self.engine.reset()

        # ----------------------------------------------
        # 清除上一轮遗落警报
        # ----------------------------------------------

        self.hide_alert()

        # ----------------------------------------------
        # 立即刷新界面
        # ----------------------------------------------

        self.update_view(
            self.engine,
            None,
        )

    # ==================================================
    # Update View
    # ==================================================

    def update_view(
        self,
        engine: DeskMemoryEngine,
        event: ForgottenEvent | None = None,
    ):

        # --------------------------------------------------
        # 保存当前 Engine
        #
        # RECALIBRATE DESK 按钮需要调用它
        # --------------------------------------------------

        self.engine = engine

        # ==================================================
        # User State
        # ==================================================

        state_name = (
            engine.person_state.state.name
        )

        if state_name == "PRESENT":

            state_color = self.green
            description = "User detected at desk"

        elif state_name == "POSSIBLE_LEAVE":

            state_color = self.yellow
            description = "Checking temporary absence"

        elif state_name == "CONFIRMED_LEAVE":

            state_color = self.red
            description = "User departure confirmed"

        else:

            state_color = self.text_secondary

            if not engine._baseline_ready:

                description = (
                    "Calibrating empty desk baseline"
                )

            else:

                description = (
                    "Waiting for user"
                )

        self.user_state_label.config(
            text=state_name,
            fg=state_color,
        )

        self.state_dot.config(
            fg=state_color
        )

        self.user_state_description.config(
            text=description
        )

        # ==================================================
        # Baseline Objects
        # ==================================================

        self._clear_container(
            self.baseline_container
        )

        baseline_items = list(
            engine.baseline.objects.values()
        )

        if not baseline_items:

            if not engine._baseline_ready:

                baseline_text = (
                    "Calibrating empty desk..."
                )

            else:

                baseline_text = (
                    "No baseline objects"
                )

            self._empty_row(
                self.baseline_container,
                baseline_text,
            )

        else:

            for item in baseline_items:

                self._add_object_row(
                    self.baseline_container,
                    item.class_name,
                )

        # ==================================================
        # Session Objects
        # ==================================================

        self._clear_container(
            self.session_container
        )

        if (
            engine.session is None
            or not engine.session.objects
        ):

            self._empty_row(
                self.session_container,
                "No session objects detected"
            )

        else:

            for item in (
                engine.session.objects.values()
            ):

                status = (
                    "PRESENT"
                    if item.visible
                    else "REMOVED"
                )

                self._add_object_row(
                    self.session_container,
                    item.class_name,
                    status,
                )

        # ==================================================
        # Forgotten Event
        # ==================================================

        if event is not None:

            names = [
                item.class_name
                for item in event.items
            ]

            self.show_alert(
                names
            )

        # 用户重新出现后清除上一轮提醒
        elif state_name == "PRESENT":

            self.hide_alert()

    # ==================================================
    # Run
    # ==================================================

    def run(self):

        self.root.mainloop()