from PyQt5.QtWidgets import (QWidget, QComboBox, QApplication,
	QGridLayout, QPushButton, QLabel, QTextEdit, QStackedWidget,
	QTabWidget, QTableView, QTableWidget, QTableWidgetItem,
	QHeaderView)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from RunApp import RunDQN
import sys
from threading import Thread
import time

class Mysignal(QObject):
	text_print=pyqtSignal(str)

class MainWindow(QWidget):

	def __init__(self, content1, content2, parent=None):
		super(MainWindow, self).__init__(parent)
		self.resize(600, 400)

		font = QFont()
		font.setPointSize(12)
		self.setFont(font)

		self.layout = QGridLayout()

		self.method_selected = []

		self.ms=Mysignal()
		self.ms.text_print.connect(self.printToGui)
		self.result_dict1={

		}

		self.result_dtype = [

		]

		self.result_couple = [
			(0,7,2),
			(2,1,2),
			(2,4,1),
			(2,6,2),
			(2,8,24)
		]

		self.result_dict_cname = {
			"3": 'newSim.ElevatorController',
			"6": 'newSim.ElevatorState',
			"0": 'newSim.Building',
		}
		self.result_dict5 = {
			"iii": 999,
			"jjj": 1000,
		}

		self.stack_select = QWidget()
		self.stack_wait = QWidget()
		self.stack_result = QWidget()

		self.init_stack_select()
		self.init_stack_wait()
		# self.init_stack_result()

		self.stackWidget = QStackedWidget()
		self.stackWidget.addWidget(self.stack_select)
		self.stackWidget.addWidget(self.stack_wait)
		# self.stackWidget.addWidget(self.stack_result)

		self.layout.addWidget(self.stackWidget)

		self.setLayout(self.layout)

	def printToGui(self,text):
		self.append_text(str(text))

	def init_stack_select(self):
		layout = QGridLayout()
		#也可以试试用QFormLayout
		label1 = QLabel("选择程序样例")
		label2 = QLabel("选择算法")
		cb1 = QComboBox(self.stack_select)
		cb2 = QComboBox(self.stack_select)
		ok = QPushButton("确  定", self.stack_select)

		cb1.addItems(content1)
		cb2.addItems(content2)

		cb1.currentIndexChanged[str].connect(self.do_select_value1)
		cb2.currentIndexChanged[str].connect(self.do_select_value2)
		ok.clicked.connect(self.stack_select_over_clicked)

		layout.addWidget(label1, 1,1)
		layout.addWidget(cb1, 1,2,1,2)
		layout.addWidget(label2, 2,1)
		layout.addWidget(cb2, 2,2,1,2)
		layout.addWidget(ok, 3,1,1,3)

		self.stack_select.setLayout(layout)
		self.selected1 = ""
		self.selected2 = ""

	def do_select_value1(self, i):
		self.selected1 = i

	def do_select_value2(self, i):
		self.selected2 = i

	def stack_select_over_clicked(self):
		self.method_selected = [self.selected1, self.selected2]
		self.stackWidget.setCurrentIndex(1)
		run = RunDQN(self.selected1, self.selected2)
		self.result_dict1,self.result_dtype,self.result_couple=run.runDQN(self.ms.text_print.emit)
		print("name\n{}".format(self.result_dict_cname))
		print(self.result_couple)
		self.init_stack_result()
		self.stackWidget.addWidget(self.stack_result)


		'''
		def target():
			run=RunDQN(self.selected1,self.selected2)
			run.runDQN(self.append_text)

		t = threading.Thread(target=target)
		t.start()
		'''


		#a,b,c=runDQN(alg,file,self.append_text)

	def init_stack_wait(self):
		self.edit = QTextEdit()
		ok = QPushButton("确  定")
		layout = QGridLayout()

		self.edit.setReadOnly(True)
		layout.addWidget(self.edit)
		layout.addWidget(ok)
		ok.clicked.connect(self.stack_wait_over_clicked)

		self.stack_wait.setLayout(layout)

		self.content = ""

	def append_text(self, text:str):
		self.content = self.content + text
		self.edit.setPlainText(self.content)

	def stack_wait_over_clicked(self):
		self.stackWidget.setCurrentIndex(2)


	def init_stack_result(self):

		layout = QGridLayout()

		tab1 = QWidget()
		tab2 = QWidget()
		tab3 = QWidget()
		tab4 = QWidget()
		tab5 = QWidget()

		tabwidget = QTabWidget(self.stack_result)
		tabwidget.addTab(tab1, "结果")
		tabwidget.addTab(tab2, "类间依赖关系")
		tabwidget.addTab(tab3, "类间耦合度")
		tabwidget.addTab(tab4, "类列表")
		tabwidget.addTab(tab5, "tab5")

		self.init_result_tab(tab1, self.result_dict1)
		self.init_result_tab_tuple(tab2, self.result_dtype, 12)
		self.init_result_tab_tuple(tab3, self.result_couple, 12)
		self.init_result_tab(tab4, self.result_dict_cname)
		self.init_result_tab(tab5, self.result_dict5)

		layout.addWidget(tabwidget)
		self.stack_result.setLayout(layout)

	def init_result_tab(self, tab, res_dict):

		layout = QGridLayout()

		tableWidget = QTableWidget()
		tableWidget.setRowCount(len(res_dict))
		tableWidget.setColumnCount(2)
		tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		tableWidget.setEditTriggers(QTableView.NoEditTriggers)

		count = 0

		for i in res_dict:
			item1 = QTableWidgetItem(i)
			item2 = QTableWidgetItem( str( res_dict[i]) )
			tableWidget.setItem(count, 0, item1)
			tableWidget.setItem(count, 1, item2)
			count += 1

		tableWidget.verticalHeader().setVisible(False)
		tableWidget.horizontalHeader().setVisible(False)

		layout.addWidget(tableWidget)

		tab.setLayout(layout)


	def init_result_tab_tuple(self, tab, res_list_of_tuple, range_of_input): #为要打印一张n*n的表，因此要知道n有多大。
		layout = QGridLayout()

		tableWidget = QTableWidget()
		tableWidget.setRowCount(range_of_input)
		tableWidget.setColumnCount(range_of_input)
		tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		tableWidget.setEditTriggers(QTableView.NoEditTriggers)

		header = []

		for i in range(0, range_of_input):
			header.append(str(i))

		tableWidget.setVerticalHeaderLabels(header)
		tableWidget.setHorizontalHeaderLabels(header)

		for (r,c,v) in res_list_of_tuple:
			item = QTableWidgetItem( str(v) )
			tableWidget.setItem( int(r), int(c), item )   #先行号，后列号

		layout.addWidget(tableWidget)

		tab.setLayout(layout)



if __name__ == "__main__":
	app = QApplication(sys.argv)
	content1 = ['ant','ATM','BCEL','daisy','DNS','SPM','simple','test']
	content2 = ["DQN", "duelingDQN", "doubleDQN", "DQN-CNN","PrioritizedReplayDQN"]
	mainwindow = MainWindow(content1, content2)
	mainwindow.show()
	app.exec_()           #有这行可以控制界面关闭后执行后面的内容，没这行则同时执行


	#sys.exit(app.exec_())    #不加这一句，可以让窗口程序和其它程序一起执行。


