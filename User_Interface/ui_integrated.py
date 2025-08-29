import os
import csv
import numpy as np
import pandas as pd
from geomdl import NURBS
from geomdl.knotvector import generate
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QMessageBox, QTextEdit
from PyQt5.QtCore import Qt, QObject, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator
import pyvista as pv
from pyvistaqt import QtInteractor
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# import ui_first
import ui_flow

os.environ['PYVISTA_USE_OFF_SCREEN'] = 'True'


# 自动检查文件夹内有无新增文件，用于自动读入时序数据
class WatchdogWorker(QObject, FileSystemEventHandler):
    """运行在 QThread 里的 watchdog 事件处理器"""
    file_created = pyqtSignal(str)  # 把新增文件路径发给主线程

    def __init__(self, root):
        super().__init__()
        self.root = root

    def on_created(self, event):
        if not event.is_directory:
            self.file_created.emit(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self.file_created.emit(event.dest_path)


class window_flow(ui_flow.Ui_Dialog, QDialog):
    def __init__(self):
        super(window_flow, self).__init__()

        self.setupUi(self)

        # 窗口上方：去除问号，保留最小化、关闭
        self.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        # ********************绘图初始化********************
        # 绘图区域，使用pyvista和pyqt的接口QtInteractor
        self.plotter = QtInteractor(self.frame_graph)
        self.plotter.set_background(color='#E3EFFC')
        self.plotter.show_axes()

        # ********************功能区：模型：油箱与管路********************
        # 油箱
        self.grid_fueltank = None
        self.actor_fueltank = None
        # stl文件默认单位mm
        self.comboBox_111.setCurrentIndex(1)
        # 是否隐藏油箱，默认显示
        self.radioButton_113.setChecked(True)

        # 关节与管道
        # 关节坐标与管道连接方式
        self.file_joint: str = 'joint_info.csv'
        self.file_tube: str = 'tube_info.csv'
        self.lineEdit_1231.setText(self.file_joint)
        self.lineEdit_1241.setText(self.file_tube)
        self.coord_joint, self.line_tube = None, None
        # 关节与管道的网格
        self.grids_joint, self.grids_tube = [], []
        self.actors_joint, self.actors_tube = [], []
        # 关节是方形的 cube，边长 self.joint_diameter 可调整，默认值 50mm
        self.joint_diameter: float = 50
        self.lineEdit_1232.setValidator(QDoubleValidator())
        self.lineEdit_1232.setText(str(self.joint_diameter))
        # 管道是圆柱形 spline，半径 self.tube_radius 可调整，默认值 20mm
        self.tube_radius: float = 20
        self.lineEdit_1242.setValidator(QDoubleValidator())
        self.lineEdit_1242.setText(str(self.tube_radius))

        # ********************功能区：数据********************
        # 用于正确识别并按顺序读取时序文件，详见 self.readDirectoryData()
        self.data_file_head: str = 'input_'
        self.data_file_time_id = []
        # 总时间步
        self.time_total = len(self.data_file_time_id)
        # 储存各关节数据：三维数组：（时间步，物理量，关节编号）
        self.data = None
        # 当前显示的时间步
        self.time_present_id: int = 0
        # 自动读取
        self.radioButton_213.setChecked(False)

        # 属性：展示速度还是压强还是温度
        self.comboBox.setCurrentIndex(0)

        # ********************watchdog检查文件夹内有无新增文件********************
        self.thread = QThread()
        self.observer = Observer()
        self.worker = None

        # ********************功能区：监控********************
        self.comboBox_311.setCurrentIndex(0)
        self.lineEdit_3111.setValidator(QDoubleValidator())
        self.lineEdit_3112.setValidator(QDoubleValidator())

        # ********************视图区********************
        # 切换不同属性时，数值条的标题和范围
        self.scalar_bar_title: list[str] = ['velocity (mm/s)', 'pressure (Pa)', 'temperature (K)']
        self.scalar_bar_unit: list[str] = ['mm/s', 'Pa', 'K']
        self.scalar_bar_range: list[list[float]] = [[15, 90], [100, 800], [283, 363]]
        self.show_bar_range()

        # ********************定时器********************
        self.timer = QTimer(self)
        self.timer.setInterval(500)  # 每500ms播放一帧
        self.timer.timeout.connect(self.show_run_next)
        self.flag_autorun: bool = False

        self.plotter.enable_cell_picking(callback=self.onPick, left_clicking=True, show_message=False, show_point=True)

        # ********************功能区：模型********************
        self.pushButton_111.clicked.connect(self.get_file_fueltank)
        self.pushButton_112.clicked.connect(self.read_file_fueltank)
        self.pushButton_121.clicked.connect(self.get_directory_tube)
        self.pushButton_122.clicked.connect(self.read_directory_tube)
        self.radioButton_113.toggled.connect(self.set_fueltank_opacity)
        self.lineEdit_1231.editingFinished.connect(self.set_file_joint)
        self.lineEdit_1241.editingFinished.connect(self.set_file_tube)
        self.lineEdit_1232.editingFinished.connect(self.set_joint_diameter)
        self.lineEdit_1242.editingFinished.connect(self.set_tube_radius)

        # ********************功能区：数据********************
        self.pushButton_211.clicked.connect(self.get_directory_data)
        self.pushButton_212.clicked.connect(self.read_directory_data)
        self.radioButton_213.toggled.connect(self.auto_read_data)
        self.pushButton_215.clicked.connect(self.clear_directory_data)

        # ********************功能区：监控********************
        self.comboBox_311.currentIndexChanged.connect(self.show_bar_range)
        self.lineEdit_3111.editingFinished.connect(self.set_bar_range_min)
        self.lineEdit_3112.editingFinished.connect(self.set_bar_range_max)
        self.pushButton_300.clicked.connect(self.clear_monitor)
        self.textBrowser_300.setLineWrapMode(QTextEdit.NoWrap)

        # ********************视图区*******************
        self.comboBox.currentIndexChanged.connect(self.show_attribute)
        self.pushButton_xy.clicked.connect(self.plotter.view_xy)
        self.pushButton_yx.clicked.connect(self.plotter.view_yx)
        self.pushButton_yz.clicked.connect(self.plotter.view_yz)
        self.pushButton_zy.clicked.connect(self.plotter.view_zy)
        self.pushButton_zx.clicked.connect(self.plotter.view_zx)
        self.pushButton_xz.clicked.connect(self.plotter.view_xz)
        self.pushButton_first.clicked.connect(self.show_first)
        self.pushButton_last.clicked.connect(self.show_last)
        self.pushButton_previous.clicked.connect(self.show_previous)
        self.pushButton_next.clicked.connect(self.show_next)
        self.pushButton_run.setChecked(False)
        self.pushButton_run.toggled.connect(self.show_run)

        # ********************说明********************
        self.pushButton_00.clicked.connect(self.readme_title)
        self.pushButton_110.clicked.connect(self.readme_fueltank)
        self.pushButton_120.clicked.connect(self.readme_tube)
        self.pushButton_210.clicked.connect(self.readme_data)
        self.pushButton_213.clicked.connect(self.readme_auto_data)
        self.pushButton_310.clicked.connect(self.readme_monitor)

        # ********************关闭页面********************
        self.pushButton_0.clicked.connect(self.cancel)

    # ************************************************************
    # ************************************************************
    # ******************************功能区：模型******************************
    # ************************************************************
    # ************************************************************
    # “选择文件”按钮：选择油箱模型文件路径
    def get_file_fueltank(self):
        path_file, file_type = QtWidgets.QFileDialog.getOpenFileName(self, caption='请选择文件', filter='stl (*.stl)')
        self.lineEdit_112.setText(path_file)

    # “读取”按钮：读取油箱模型文件
    def read_file_fueltank(self):
        # 如果已经读取过油箱模型文件，询问是否覆盖
        if self.grid_fueltank is not None:
            if QMessageBox.No == QMessageBox.question(self, '温馨提示', '已读取过油箱模型，是否覆盖？',
                                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
                return
        path_file_fueltank = self.lineEdit_112.text().strip()
        # 路径必须存在且必须是stl文件
        if os.path.exists(path_file_fueltank) and path_file_fueltank.lower().endswith('.stl'):
            # stl文件必须能被正常读取
            try:
                self.grid_fueltank = pv.read(path_file_fueltank)
                # 单位m，读进来改成mm
                if int(self.comboBox_111.currentIndex()) == 0:
                    self.grid_fueltank.points *= 1000
            except Exception as error:
                QtWidgets.QMessageBox.critical(None, '温馨提示', '文件读取失败，请检查文件是否损坏！')
                return
            # 读取完毕，显示油箱
            self.show_fueltank_init()
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '文件不存在或类型不正确，请检查路径是否正确！')

    # “选择文件夹”按钮：选择管路文件夹路径
    def get_directory_tube(self):
        self.lineEdit_122.setText(QtWidgets.QFileDialog.getExistingDirectory(self, caption='请选择文件夹'))

    def read_directory_tube(self):
        # 如果已经读取过管路文件，询问是否覆盖
        if (self.coord_joint is not None) and (self.line_tube is not None):
            if QMessageBox.No == QMessageBox.question(self, '温馨提示', '已读取过管路文件，是否覆盖？',
                                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
                return
        path_folder = self.lineEdit_122.text().strip()
        # 检查文件夹是否存在
        if os.path.exists(path_folder):
            # 读取关节坐标
            path_joint = os.path.join(path_folder, self.file_joint)
            if os.path.exists(path_joint):
                try:
                    self.coord_joint = pd.read_csv(path_joint, header=0, encoding='utf-8').fillna(0).to_numpy()[:, 1:4]
                    self.set_grids_joint()
                except Exception as error:
                    QtWidgets.QMessageBox.critical(None, '温馨提示', '{} 读取失败，请检查文件格式是否正确！'.format(
                        self.file_joint))
                    return
            else:
                QtWidgets.QMessageBox.critical(None, '温馨提示', '文件夹内不存在 {}，请检查文件名是否正确！'.format(
                    self.file_joint))
                return
            # 读取管道连接方式
            path_tube = os.path.join(path_folder, self.file_tube)
            if os.path.exists(path_tube):
                try:
                    # 需要支持列数不定的csv，不能用pandas
                    line_tube = []
                    with open(path_tube, newline='', encoding='utf-8') as f:
                        f_reader = csv.reader(f)
                        next(f_reader)  # 跳过首行标题行
                        for f_row in f_reader:  # 逐行读取
                            # 列数必然是3的倍数（列依次为：管道编号、管道连接的2个关节编号、中间n个NURBS形状控制点的3维坐标）
                            if len(f_row) % 3 == 0:
                                line_tube.append(list(map(float, f_row)))  # 把每行的字符串转 float
                    if line_tube:
                        self.line_tube = line_tube
                        self.set_grids_tube_spline()
                except Exception as error:
                    QtWidgets.QMessageBox.critical(None, '温馨提示', '{} 读取失败，请检查文件格式是否正确！'.format(
                        self.file_tube))
                    return
            else:
                QtWidgets.QMessageBox.critical(None, '温馨提示', '文件夹内不存在 {}，请检查文件名是否正确！'.format(
                    self.file_tube))
                return
            # 读取完毕，显示管路
            self.show_tube_init()
            self.show_tube_data_time()
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '文件夹不存在，请检查路径是否正确！')

    def set_file_joint(self):
        self.file_joint = self.lineEdit_1231.text().strip()

    def set_file_tube(self):
        self.file_tube = self.lineEdit_1241.text().strip()

    def set_joint_diameter(self):
        joint_diameter = float(self.lineEdit_1232.text().strip())
        if joint_diameter <= 0:
            self.joint_diameter = 50
            self.lineEdit_1232.setText(str(self.joint_diameter))
        else:
            self.joint_diameter = joint_diameter
        self.set_grids_joint()
        self.show_tube_init()
        self.show_tube_data_time()

    def set_tube_radius(self):
        tube_radius = float(self.lineEdit_1242.text().strip())
        if tube_radius <= 0:
            self.tube_radius = 20
            self.lineEdit_1242.setText(str(self.tube_radius))
        else:
            self.tube_radius = tube_radius
        self.set_grids_tube_spline()
        self.show_tube_init()
        self.show_tube_data_time()

    # 生成关节网格
    def set_grids_joint(self):
        if self.coord_joint is None:
            return
        self.grids_joint = []
        for i in range(self.coord_joint.shape[0]):
            try:
                # 关节是方形的 cube，边长 self.joint_diameter 可调整
                grid_joint = pv.Cube(center=self.coord_joint[i], x_length=self.joint_diameter,
                                     y_length=self.joint_diameter, z_length=self.joint_diameter)
                grid_joint['data'] = np.zeros(grid_joint.GetNumberOfPoints())
                self.grids_joint.append(grid_joint)
            except Exception as error:
                continue

    # 生成管道网格
    def set_grids_tube_spline(self):
        if (self.coord_joint is None) or (self.line_tube is None):
            return
        self.grids_tube = []
        for row_tube in self.line_tube:
            num_col_tube = len(row_tube)
            try:
                if (num_col_tube < 3) or (not num_col_tube % 3 == 0):
                    continue
                elif num_col_tube == 3:
                    ctrl_points = np.array([self.coord_joint[int(row_tube[1])],
                                            self.coord_joint[int(row_tube[2])]])
                else:
                    ctrl_points = np.array([self.coord_joint[int(row_tube[1])],
                                            *[row_tube[i:i + 3] for i in range(3, num_col_tube, 3)],
                                            self.coord_joint[int(row_tube[2])]])
                # 使用 geomdl
                curve = NURBS.Curve()
                curve.degree = min(3, len(ctrl_points) - 1)  # 必须满足 degree ≤ (控制点数 − 1)，但是 degree 越大计算量越大
                curve.ctrlpts = ctrl_points
                curve.knotvector = generate(curve.degree, len(curve.ctrlpts))  # 自动计算合理的节点向量
                curve.delta = 0.01  # 取样密度，可调整，0.01约等于100段
                points = np.array(curve.evalpts)  # 得到曲线上的离散点
                # 使用 pyvista
                poly = pv.PolyData()
                poly.points = points
                poly.lines = np.hstack([np.array([len(points)] + list(range(len(points))), dtype=np.int32)])
                # 管道是圆柱形 spline，半径 self.tube_radius 可调整
                grid_tube = poly.tube(radius=self.tube_radius)
                grid_tube['data'] = np.zeros(grid_tube.GetNumberOfPoints())
                self.grids_tube.append(grid_tube)
            except Exception as error:
                continue

    # ************************************************************
    # ************************************************************
    # ******************************功能区：数据******************************
    # ************************************************************
    # ************************************************************
    # “选择文件”按钮：选择数据文件：速度、压强、温度
    def get_directory_data(self):
        self.lineEdit_212.setText(QtWidgets.QFileDialog.getExistingDirectory(self, caption='请选择文件夹'))

    # “读取”按钮：读取数据文件：速度、压强、温度
    def read_directory_data(self):
        # 如果未读取过管路文件，则需要先读取管路文件
        if (self.coord_joint is None) or (self.line_tube is None):
            QtWidgets.QMessageBox.critical(None, '温馨提示', '请先读取管路文件！')
            return
        # 如果已经读取过数据文件，询问是否覆盖
        if self.data is not None:
            if QMessageBox.No == QMessageBox.question(self, '温馨提示', '已读取过数据文件，是否覆盖？',
                                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
                return
        self.clear_data()
        path_folder = self.lineEdit_212.text().strip()
        # 从一个文件夹内读取多个csv文件，csv里储存有各关节的速度、压强、温度
        if os.path.exists(path_folder):
            # 第一步：寻找需要的csv文件（若文件夹名为 input，则该文件夹下名为 input_xx.csv 的csv文件是需要的）
            # 例如：若 input 文件夹下存在4个文件：input_11.csv，output_3.csv，input_4.csv，not_input_6.csv
            # 则第一步完成后，self.data_file_time_id 的值为 [11, 4]
            self.data_file_head = os.path.basename(path_folder) + '_'
            self.data_file_time_id = []
            for name_file in os.listdir(path_folder):
                if name_file.startswith(self.data_file_head) and name_file.endswith('.csv'):
                    name_id = name_file.lstrip(self.data_file_head).rstrip('.csv')
                    if name_id.isdigit():
                        self.data_file_time_id.append(int(name_id))

            # 第二步：把需要的csv文件排序
            # 例如：第一步可能先发现了 input_11.csv，再发现了 input_4.csv，但按时间顺序应先读取 input_4.csv，再读取 input_11.csv
            # 若第二步完成前，self.data_file_time_id 的值为 [11, 4]，则第二步完成后，self.data_file_time_id 的值为 [4, 11]
            self.data_file_time_id.sort()

            # 第三步：依顺序读取csv（读取各时间步文件，并显示读取情况）
            # 例如：若 input_4.csv 读取正确，input_11.csv 读取错误
            # 则第三步完成后，self.data_file_time_id 的值为 [4]，self.time_total 为 1
            # 并显示：
            # input_4.csv  success
            # input_11.csv  fail ! ! ! !
            text_read = ''
            self.data = []
            for name_id in self.data_file_time_id:
                name_file = self.data_file_head + str(name_id) + '.csv'
                path_file = os.path.join(path_folder, name_file)
                try:
                    data_t = pd.read_csv(path_file, header=0, encoding='utf-8').fillna(0).to_numpy()[:, 1:4]
                    # 转置，按（物理量，关节编号）的顺序
                    self.data.append(data_t.T)
                    text_read += name_file + '  success\n'
                except Exception as error:
                    self.data_file_time_id.remove(name_id)
                    text_read += name_file + '  fail ! ! ! !\n'
            self.data = np.array(self.data)
            self.time_total = len(self.data_file_time_id)

            if not self.data_file_time_id:
                text_read = 'no file !'

            self.textBrowser_215.setText(text_read)

            # 绘制第一个时间步
            if self.data is not None:
                self.time_present_id = 0
                self.show_tube_data_time()

        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '文件夹不存在，请检查路径是否正确！')

    # “清空”按钮：清空数据文件：速度、压强、温度
    def clear_directory_data(self):
        if self.data is not None:
            if QMessageBox.Yes == QMessageBox.question(self, '温馨提示', '确认清空已读过的数据？',
                                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
                self.clear_data()
        else:
            QtWidgets.QMessageBox.critical(None, '温馨提示', '无数据！')

    def clear_data(self):
        # 清空数据
        self.data_file_time_id = []
        self.time_total = len(self.data_file_time_id)
        self.data = None
        self.textBrowser_215.setText('')
        self.textBrowser_300.setText('')
        self.show_not_data()

    def show_not_data(self):
        # 清空时间步
        self.time_present_id = 0
        self.textBrowser_2.setText('?')
        self.textBrowser_3001.setText('?')
        # 重新绘制
        self.set_mapper_mode(False)
        if len(self.plotter.scalar_bars):
            self.plotter.remove_scalar_bar()

    def get_new_data(self, path_file: str):
        if os.path.exists(path_file):
            name_file = os.path.basename(path_file)
            if name_file.startswith(self.data_file_head) and name_file.endswith('.csv'):
                name_id = name_file.lstrip(self.data_file_head).rstrip('.csv')
                if name_id.isdigit():
                    try:
                        data_t = pd.read_csv(path_file, header=0, encoding='utf-8').fillna(0).to_numpy()[:, 1:4]
                        text_read = self.textBrowser_215.toPlainText()
                        text_read += name_file + '  success\n'
                        self.textBrowser_215.setText(text_read)
                    except Exception as error:
                        text_read = self.textBrowser_215.toPlainText()
                        text_read += name_file + '  fail ! ! ! !\n'
                        self.textBrowser_215.setText(text_read)
                        return
                    data_t = np.expand_dims(data_t.T, axis=0)
                    if self.data is None:
                        self.data = data_t
                    else:
                        self.data = np.concatenate([self.data, data_t], axis=0)
                    self.data_file_time_id.append(int(name_id))
                    self.time_total = len(self.data_file_time_id)
                    self.show_last()

    def auto_read_data(self):
        if not self.radioButton_213.isChecked():
            self.clear_observer()
            return
        # 如果未读取过管路文件，则需要先读取管路文件
        if (self.coord_joint is None) or (self.line_tube is None):
            QtWidgets.QMessageBox.critical(None, '温馨提示', '请先读取管路文件！')
            return
        path_folder = self.lineEdit_212.text().strip()
        # 从一个文件夹内读取多个csv文件，csv里储存有各关节的速度、压强、温度
        if os.path.exists(path_folder):
            self.worker = WatchdogWorker(path_folder)
            self.worker.moveToThread(self.thread)
            self.observer.schedule(self.worker, path_folder, recursive=False)

            self.thread.started.connect(lambda: self.observer.start())
            self.thread.start()
            self.worker.file_created.connect(self.get_new_data)

    def clear_observer(self):
        # 取消所有监听
        self.observer.unschedule_all()
        # 停 watchdog（这步必须在 wait 之前，否则 observer 线程阻塞）
        self.observer.stop()
        # 千万不要让 GUI 直接调用 observer.join()，否则会卡死
        # 直接不需要写 self.observer.join()

        # 让 QThread 事件循环结束
        # 发送退出信号
        self.thread.quit()
        # 阻塞等待线程真正结束
        self.thread.wait()

    # ************************************************************
    # ************************************************************
    # ******************************功能区：监控******************************
    # ************************************************************
    # ************************************************************
    def set_bar_range_min(self):
        id_attr = int(self.comboBox_311.currentIndex())
        self.scalar_bar_range[id_attr][0] = float(self.lineEdit_3111.text().strip())
        self.show_tube_data_time()

    def set_bar_range_max(self):
        id_attr = int(self.comboBox_311.currentIndex())
        self.scalar_bar_range[id_attr][1] = float(self.lineEdit_3112.text().strip())
        self.show_tube_data_time()

    def show_monitor_data(self, id_attr: int, data_now):
        txt = self.textBrowser_300.toPlainText()
        if txt == '':
            txt = '关节\t' + 'time_step\t' + 'attribute\t' + 'location (mm)\t\n'
        for i in range(len(self.grids_joint)):
            if not float(self.scalar_bar_range[id_attr][0]) <= data_now[i] <= float(self.scalar_bar_range[id_attr][1]):
                txt += ('joint_' + str(i) + '\t' + str(self.data_file_time_id[int(self.time_present_id)]) + '\t' +
                        '{:.2f} '.format(data_now[i]) + self.scalar_bar_unit[id_attr] + '\t' + str(
                            np.round(self.grids_joint[i].center, 2)) + '\n')
        self.textBrowser_300.setText(txt)

    def clear_monitor(self):
        if QMessageBox.Yes == QMessageBox.question(self, '温馨提示', '确认清空监控区的记录？',
                                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
            self.textBrowser_300.setText('')

    # ************************************************************
    # ************************************************************
    # ******************************视图区******************************
    # ************************************************************
    # ************************************************************
    def show_fueltank_init(self):
        # 先关闭自动播放状态
        self.flag_autorun = False
        self.remove_actor_fueltank()
        if self.grid_fueltank is not None:
            self.actor_fueltank = self.plotter.add_mesh(self.grid_fueltank, name='fueltank', color='gainsboro',
                                                        show_edges=False, show_scalar_bar=False, opacity=1,
                                                        pickable=True)

    def set_fueltank_opacity(self):
        if self.actor_fueltank is not None:
            self.actor_fueltank.GetProperty().SetOpacity(1 if self.radioButton_113.isChecked() else 0)

    def remove_actor_fueltank(self):
        if self.actor_fueltank is not None:
            self.plotter.remove_actor(self.actor_fueltank)
        self.actor_fueltank = None

    def show_tube_init(self):
        # 先关闭自动播放状态
        self.flag_autorun = False
        self.remove_actors_joint()
        self.remove_actors_tube()
        for i in range(len(self.grids_joint)):
            self.actors_joint.append(
                self.plotter.add_mesh(self.grids_joint[i], name='joint_' + str(i), color=pv.global_theme.color,
                                      cmap='jet', show_edges=True, line_width=1, show_scalar_bar=False, pickable=True))
        for i in range(len(self.grids_tube)):
            self.actors_tube.append(
                self.plotter.add_mesh(self.grids_tube[i], name='tube_' + str(i), color=pv.global_theme.color,
                                      cmap='jet', show_edges=False, show_scalar_bar=False, pickable=True))

    def set_mapper_mode(self, flag: bool):
        if flag:
            for i in range(len(self.actors_joint)):
                self.actors_joint[i].mapper.SetScalarVisibility(True)
            for i in range(len(self.actors_tube)):
                self.actors_tube[i].mapper.SetScalarVisibility(True)
        else:
            for i in range(len(self.actors_joint)):
                self.actors_joint[i].mapper.SetScalarVisibility(False)
            for i in range(len(self.actors_tube)):
                self.actors_tube[i].mapper.SetScalarVisibility(False)

    def remove_actors_joint(self):
        if self.actors_joint:
            for actor_joint in self.actors_joint:
                self.plotter.remove_actor(actor_joint)
        self.actors_joint = []

    def remove_actors_tube(self):
        if self.actors_tube:
            for actor_tube in self.actors_tube:
                self.plotter.remove_actor(actor_tube)
        self.actors_tube = []

    def show_tube_data_time(self):
        # 如果连数据都没有就无法更新数据
        if (self.data is None) or (not self.grids_joint) or (not self.grids_joint) or (not self.actors_joint) or (
                not self.actors_tube) or (len(self.data) <= int(self.time_present_id)):
            return

        # 显示当前时间步
        try:
            self.textBrowser_2.setText(str(self.data_file_time_id[int(self.time_present_id)]))
            self.textBrowser_3001.setText(str(self.data_file_time_id[int(self.time_present_id)]))
        except Exception as error:
            self.textBrowser_2.setText('?')
            self.textBrowser_3001.setText('?')
            return
        # 属性：显示速度还是压强还是温度
        id_attr = int(self.comboBox.currentIndex())

        self.set_mapper_mode(True)

        # 提取该时刻该属性的数据为 data_now，其维度等于关节数
        # 若提取不足或出错，则 data_now 对应项置零
        try:
            data_now = np.zeros(len(self.grids_joint))
            data_extract = self.data[self.time_present_id, id_attr, :]
            dim_extract = min(len(self.grids_joint), len(data_extract))
            data_now[0:dim_extract] = data_extract[0:dim_extract]
        except Exception as error:
            data_now = np.zeros(len(self.grids_joint))

        # 更新关节值
        for i in range(len(self.grids_joint)):
            self.grids_joint[i]['data'] = np.full(self.grids_joint[i].GetNumberOfPoints(),
                                                  data_now[i])
            self.grids_joint[i].Modified()
            self.actors_joint[i].mapper.SetScalarRange(self.scalar_bar_range[id_attr][0],
                                                       self.scalar_bar_range[id_attr][1])
            self.actors_joint[i].mapper.Update()
        for i in range(len(self.grids_tube)):
            self.grids_tube[i]['data'] = np.linspace(start=data_now[int(self.line_tube[i][1])],
                                                     stop=data_now[int(self.line_tube[i][2])],
                                                     num=self.grids_tube[i].GetNumberOfPoints())
            self.grids_tube[i].Modified()
            self.actors_tube[i].mapper.SetScalarRange(self.scalar_bar_range[id_attr][0],
                                                      self.scalar_bar_range[id_attr][1])
            self.actors_tube[i].mapper.Update()
        self.plotter.render()

        # 绘制 scalar_bar
        # self.plotter.scalar_bars 储存当前画板上的 scalar_bar 信息
        # 若不存在 scalar_bar，则 len(self.plotter.scalar_bars) = 0，此时无需清除 scalar_bar
        if len(self.plotter.scalar_bars):
            for name, scalar_bar in list(self.plotter.scalar_bars.items()):
                if not scalar_bar.GetTitle() == self.scalar_bar_title[id_attr]:
                    self.plotter.remove_scalar_bar(name)
        if not self.plotter.scalar_bars:
            self.plotter.add_scalar_bar(title=self.scalar_bar_title[id_attr], color='firebrick',
                                        title_font_size=15, label_font_size=10, width=0.5,
                                        vertical=False, position_x=0.3, position_y=0.1)

        self.show_monitor_data(id_attr, data_now)

        if not self.comboBox_311.currentIndex() == id_attr:
            self.comboBox_311.setCurrentIndex(id_attr)

    def show_attribute(self):
        # 先关闭自动播放状态
        self.flag_autorun = False
        self.show_tube_data_time()

    def show_first(self):
        self.flag_autorun = False
        if self.time_present_id == 0:
            return
        else:
            self.time_present_id = 0
            self.show_tube_data_time()

    def show_last(self):
        self.flag_autorun = False
        if self.time_present_id == max(int(self.time_total - 1), 0):
            return
        else:
            self.time_present_id = max(int(self.time_total - 1), 0)
            self.show_tube_data_time()

    def show_previous(self):
        self.flag_autorun = False
        if self.time_present_id == 0:
            return
        else:
            self.time_present_id = max(int(self.time_present_id - 1), 0)
            self.show_tube_data_time()

    def show_next(self):
        self.flag_autorun = False
        if self.time_present_id == max(int(self.time_total - 1), 0):
            return
        else:
            self.time_present_id = min(int(self.time_present_id + 1), int(self.time_total - 1))
            self.show_tube_data_time()

    def show_run_next(self):
        if self.flag_autorun:
            if self.time_present_id == max(int(self.time_total - 1), 0):
                self.pushButton_run.setChecked(False)
                return
            else:
                self.time_present_id = min(int(self.time_present_id + 1), int(self.time_total - 1))
                self.show_tube_data_time()
        else:
            self.pushButton_run.setChecked(False)

    def show_run(self, flag_check: bool):
        self.flag_autorun = flag_check
        if self.flag_autorun:
            self.timer.start()
        else:
            self.timer.stop()

    def onPick(self, picked_mesh, cid):
        if picked_mesh is None or cid == -1:
            return

        # 取出 scalar
        try:
            scalar_val = picked_mesh['data'][cid]
        except Exception as error:
            scalar_val = 0
        print(cid, scalar_val)

        # 取出 cell 中心坐标（用于标注）
        center = picked_mesh.cell_centers().points[cid]

        # 打标签
        self.plotter.add_point_labels(
            [center],
            [f"{scalar_val:.2f}"],
            font_size=15,
            text_color='white',
            point_color='yellow',
            point_size=12,
            always_visible=True
        )

    def show_bar_range(self):
        id_attr = int(self.comboBox_311.currentIndex())
        self.lineEdit_3111.setText(str(self.scalar_bar_range[id_attr][0]))
        self.lineEdit_3112.setText(str(self.scalar_bar_range[id_attr][1]))

        if not self.comboBox.currentIndex() == id_attr:
            self.comboBox.setCurrentIndex(id_attr)

    # ************************************************************
    # ************************************************************
    # ******************************说明******************************
    # ************************************************************
    # ************************************************************
    @staticmethod
    def readme_title():
        txt_readme = ('中国航空工业集团公司沈阳飞机设计研究所\n\n'
                      '燃油分系统典型子系统数字孪生模型\n'
                      '本软件支持以下功能：\n\n'
                      '1、读取油箱和管路模型。\n'
                      '2、读取流体仿真的过程参数，作为数据输入。\n'
                      '3、根据流体仿真的过程数据，连续动态显示管路端口的速度、压强、温度信息。\n'
                      '4、可设定参数有效范围，监控参数是否超出范围。\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    @staticmethod
    def readme_fueltank():
        txt_readme = ('功能区：模型 - ①油箱\n'
                      '使用说明：\n\n'
                      '用于导入油箱的几何模型\n\n'
                      '(1) 支持STL格式文件导入，由于STL格式不包括单位，导入前请先设置长度单位。\n'
                      '1、点击”选择文件(.stl)“，选择读取的STL文件。\n'
                      '2、在”文件单位“选择待读取的STL文件的长度单位（m or mm）。\n'
                      '3、点击”读取“，根据路径读取STL文件并显示。\n\n'
                      '(2) ”显示油箱“可随时切换油箱状态为显示或隐藏。\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    @staticmethod
    def readme_tube():
        txt_readme = ('功能区：模型 - ②管路\n'
                      '使用说明：\n\n'
                      '用于导入管路的几何模型\n\n'
                      '需要读取2个csv文件：\n\n'
                      '(1) 关节坐标文件：\n'
                      '1、默认名：joint_info.csv，文件名可随时调整。\n'
                      '2、格式：csv，[1 + num_joint, 4]，utf-8。\n'
                      '  行：行数为：1 + 关节数；首行为标题行，不读取\n'
                      '  列：列数为：4；依次为：关节编号（从0开始）、关节x坐标（mm）、关节y坐标（mm）、关节z坐标（mm）\n'
                      '3、关节默认用小正方体绘制，小正方体的边长可随时调整。\n\n'
                      '(2) 管道连接文件：\n'
                      '1、默认名：tube_info.csv，文件名可随时调整。\n'
                      '2、格式：csv，[1 + num_tube, 3 * (n + 1)]，utf-8。\n'
                      '  行：行数为：1 + 管道数；首行为标题行，不读取\n'
                      '  列：列数为：3 的倍数；依次为：管道编号（从0开始）、管道连接的2个关节编号、中间n个NURBS形状控制点的3维坐标（mm）\n'
                      '3、管道默认用柱体绘制，柱体的半径可随时调整。\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    @staticmethod
    def readme_data():
        txt_readme = ('功能区：数据\n'
                      '使用说明：\n\n'
                      '用于读取关节数据：速度、压强、温度\n\n'
                      '(1) 有效的数据文件名：foldername_number.csv\n'
                      'foldername 是文件夹名，number 必须是数字\n'
                      '举例：\n'
                      '文件夹名：input，则该文件夹下名如 input_0.csv input_6.csv 等可被读取，input_t.csv 不可被读取。\n\n'
                      '(2) 数据文件格式：csv，[1 + num_joint, 4]，utf-8。\n'
                      '  行：行数为：1 + 关节数；首行为标题行，不读取\n'
                      '  列：列数为：4；依次为：关节编号（从0开始）、速度（mm/s）、压强（Pa）、温度（K）\n\n'
                      '(3) 数据文件出现异常值：\n'
                      '若数据csv文件里出现NaN，则该时间步默认用0替代出错的属性值。\n\n'
                      '(4) 支持两种方式读入：\n'
                      '1、一次性读取：依次点击”选择文件夹“、“读取”，把文件夹下的所有有效的数据文件读入。\n'
                      '2、自动监测读取：选中“自动监测新增文件”，每当文件夹下新增有效的数据文件时，自动读入。\n\n'
                      '(5) 清空数据：\n'
                      '点击“清空”，将清空已读入的所有数据。\n'
                      '注意：清空并不改变自动监测状态，自动监测状态只能手动更改。\n'
                      )
        QMessageBox.information(None, '使用说明', txt_readme)

    @staticmethod
    def readme_auto_data():
        txt_readme = ('功能区：数据 - 自动监测新增文件\n'
                      '使用说明：\n\n'
                      '用于自动监测读取数据文件\n\n'
                      '选中“自动监测新增文件”，每当文件夹下新增有效的数据文件时，自动读入。\n\n'
                      '1、只有当新增文件的文件名符合要求，才会被读取（foldername_number.csv）。\n'
                      '2、在该文件夹下新建文件、复制文件到该文件夹下，都被视为“新增文件”。\n'
                      '3、注意只是“新增”，文件夹下原已有的文件不算，原已有的文件请使用上方的”读取“一次性读取。\n'
                      '4、“清空”并不改变自动监测状态，自动监测状态只能手动更改。\n'
                      '5、新增文件的监测直接采用操作系统的实时文件系统事件接口，使用 watchdog 包实现。\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    @staticmethod
    def readme_monitor():
        txt_readme = ('功能区：监控\n'
                      '使用说明：\n\n'
                      '用于监控数据是否超范围\n\n'
                      '(1) 可随时设置有效范围，设置后视图区绘制范围也将立即更新。\n\n'
                      '(2) 当有关节数据超出有效范围时，在下方记录。\n')
        QMessageBox.information(None, '使用说明', txt_readme)

    # ************************************************************
    # ************************************************************
    # ******************************返回******************************
    # ************************************************************
    # ************************************************************
    # 返回按钮
    # 安全关闭
    def safe_close(self):
        # 停止自动播放
        self.pushButton_run.setChecked(False)

        # 停 watchdog 和 QThread
        self.clear_observer()

        # 必要：关闭绘图工具！
        self.plotter.close()

    def cancel(self):
        # self.close() 会调用 self.closeEvent()
        self.close()

    # 重写关闭事件
    def closeEvent(self, event):
        self.safe_close()
        event.accept()

# ********************************************************************************
# ****                                                                        ****
# ****                                 主界面                                  ****
# ****                                                                        ****
# ********************************************************************************
# class window_first(ui_first.Ui_Dialog, QDialog):
#     def __init__(self):
#         super(window_first, self).__init__()
#         self.setupUi(self)
#
#         # 窗口上方：去除问号，保留最小化、关闭
#         self.setWindowFlags(Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
#
#         # ********************设置按钮********************
#         self.pushButton_1.clicked.connect(self.runFlow)
#
#     def runFlow(self):
#         dialog_flow = window_flow()
#         dialog_flow.show()
