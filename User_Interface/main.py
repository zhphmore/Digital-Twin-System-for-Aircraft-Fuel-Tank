import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

import ui_integrated

if __name__ == '__main__':
    # 支持高分屏自动缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # 运行程序
    app = QApplication(sys.argv)
    # 显示主界面
    # app_wins = ui_integrated.window_first()
    app_wins = ui_integrated.window_flow()
    app_wins.show()
    # 关闭程序，释放资源
    sys.exit(app.exec_())
