from PyQt5.QtWidgets import (QWidget, QComboBox, QApplication,
	QGridLayout, QPushButton, QLabel, QTextEdit, QStackedWidget,
	QTabWidget, QTableView, QTableWidget, QTableWidgetItem,
	QHeaderView, QFrame, QSplitter)
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, Qt
import sys
import time
from RunApp import RunDQN

class Out_for_test(QObject):
	finished = pyqtSignal()
	returnRes = pyqtSignal(list)
	strReady = pyqtSignal(str)

	def __init__(self, s1, s2):
		super().__init__()
		self.selected1 = s1
		self.selected2 = s2

	@pyqtSlot()
	def out(self):
		#self.stackWidget.setCurrentIndex(1)
		run = RunDQN(self.selected1, self.selected2)
		self.result_dict1,self.result_dtype,self.result_couple,self.result_dict_cname=run.runDQN(self.strReady.emit)
		n=len(self.result_dict_cname)
		# print("name\n{}".format(self.result_dict_cname))
		# print(self.result_couple)

		# self.init_stack_result()
		# splitter1.addWidget(self.stack_result)

		# self.test.intReady.connect(self.append_text)
		# self.test.moveToThread(self.thread)
		# self.test.finished.connect(self.thread.quit)
		# self.thread.started.connect(self.test.out)
		#
		# self.thread.start()

		res = []
		res.append(self.result_dict1)
		res.append(self.result_dtype)
		res.append(self.result_couple)
		res.append(self.result_dict_cname)

		self.returnRes.emit(res)
		self.finished.emit()


class Mysignal(QObject):
	text_print=pyqtSignal(str)

class MainWindow(QWidget):

	def __init__(self, content1, content2, parent=None):
		super(MainWindow, self).__init__(parent)
		self.resize(1200, 800)

		font = QFont()
		font.setPointSize(12)
		self.setFont(font)

		self.layout = QGridLayout()

		self.method_selected = []
		self.result_dict1={
			"aaa": 111,
			"bbb": 222,
			"ccc": 333,
			"ddd": 444,
			"eee": 555
		}

		self.result_dtype = [
			(0,3,1.23123),
			(0,6,1.23123),
			(0,7,2.23123),
			(2,3,2.23123),
			(2,6,1.23123),
			(3,8,1.23123)
		]

		self.result_couple = [
			(0,7,2.23123),
			(2,1,2.23123),
			(2,4,1.23123),
			(2,6,2.23123),
			(2,8,24.23123)
		]

		self.result_dict_cname = {
			"fff": 666,
			"ggg": 777,
			"hhh": 888,
		}
		self.result_dict5 = {
			"iii": 999,
			"jjj": 1000,
		}

		self.table_list = []

		self.stack_select = QWidget()
		self.stack_wait = QWidget()
		self.stack_result = QWidget()

		self.init_stack_select()  #select
		self.init_stack_wait() #wait
		self.init_stack_result()

		'''

		self.stackWidget = QStackedWidget()
		self.stackWidget.addWidget(self.stack_select)
		self.stackWidget.addWidget(self.stack_wait)
		self.stackWidget.addWidget(self.stack_result)

		self.layout.addWidget(self.stackWidget)

		'''

		self.frame_top_left = QFrame()
		self.frame_top_right = QFrame()
		self.frame_bottom = QFrame()

		self.frame_top_left.setFrameShape(QFrame.StyledPanel)
		self.frame_top_right.setFrameShape(QFrame.StyledPanel)
		self.frame_bottom.setFrameShape(QFrame.StyledPanel)

		splitter1 = QSplitter(Qt.Horizontal)
		splitter1.addWidget(self.stack_select)
		splitter1.addWidget(self.stack_result)

		splitter2 = QSplitter(Qt.Vertical)
		splitter2.addWidget(splitter1)
		splitter2.addWidget(self.stack_wait)


		self.layout.addWidget(splitter2)

		self.setLayout(self.layout)





		self.thread = QThread()

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

	def printToGui(self,text):
		self.append_text(str(text))

	def do_select_value1(self, i):
		self.selected1 = i

	def do_select_value2(self, i):
		self.selected2 = i

	def stack_select_over_clicked(self):
		#self.stackWidget.setCurrentIndex(1)
		# run = RunDQN(self.selected1, self.selected2)
		# self.result_dict1,self.result_dtype,self.result_couple,self.result_dict_cname=run.runDQN(self.ms.text_print.emit)
		# n=len(self.result_dict_cname)
		# print("name\n{}".format(self.result_dict_cname))
		# print(self.result_couple)

		# self.init_stack_result()
		# splitter1.addWidget(self.stack_result)

		self.test = Out_for_test(self.selected1, self.selected2)
		self.test.strReady.connect(self.append_text)
		self.test.moveToThread(self.thread)
		self.test.finished.connect(self.thread.quit)
		self.thread.started.connect(self.test.out)
		self.test.returnRes.connect(self.getRes)

		self.thread.start()

		test1 = {
			'a':1,
			'b':2,
			'c':3,
			'd':4,
		}
		#
		# test2 = [
		# 	(1,1,1),
		# 	(2,2,2),
		# 	(3,3,3),
		# 	(4,4,4)
		# ]
		#
		# test3 = [
		# 	(5,5,5),
		# 	(6,6,6),
		# 	(7,7,7),
		# 	(8,8,8)
		# ]

	def getRes(self, res):
		n=len(res[3])
		self.update_result(0, res[0])
		self.update_result_tuple(1, res[1], n)
		self.update_result_tuple(2, res[2], n)
		self.update_result(3, res[3])
		#self.update_result(4, test1)



		#a,b,c=runDQN(alg,file,self.append_text)

	def init_stack_wait(self):
		self.edit = QTextEdit()
		ok = QPushButton("清  除")
		layout = QGridLayout()

		self.edit.setReadOnly(True)
		layout.addWidget(self.edit)
		layout.addWidget(ok)
		ok.clicked.connect(self.stack_wait_clean_clicked)

		self.stack_wait.setLayout(layout)

		self.content = ""

	def append_text(self, text:str):
		'''
		self.content = self.content + text
		self.edit.setPlainText(self.content)
		'''

		self.edit.moveCursor(QTextCursor.End)
		self.edit.append(text)

	def stack_wait_clean_clicked(self):
		#self.stackWidget.setCurrentIndex(2)
		self.edit.setPlainText("")

		#debug
		self.update_result(0, self.result_dict1)
		self.update_result_tuple(1, self.result_dtype, 12)
		self.update_result_tuple(2, self.result_couple, 12)
		self.update_result(3, self.result_dict_cname)
		self.update_result(4, self.result_dict5)


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

		'''

		self.init_result_tab(tab1, self.result_dict1)
		self.init_result_tab_tuple(tab2, self.result_tuple2, 12)
		self.init_result_tab_tuple(tab3, self.result_tuple3, 12)
		self.init_result_tab(tab4, self.result_dict4)
		self.init_result_tab(tab5, self.result_dict5)

		'''

		self.init_result_tab(tab1)
		self.init_result_tab(tab2)
		self.init_result_tab(tab3)
		self.init_result_tab(tab4)
		self.init_result_tab(tab5)

		layout.addWidget(tabwidget)
		self.stack_result.setLayout(layout)

	def init_result_tab(self, tab):

		layout = QGridLayout()


		'''
		tableWidget = QTableWidget()
		tableWidget.setRowCount(len(res_dict))
		tableWidget.setColumnCount(2)
		tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		tableWidget.setEditTriggers(QTableView.NoEditTriggers)
		'''


		self.table_list.append(QTableWidget())

		self.table_list[-1] = QTableWidget()
		self.table_list[-1].horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		self.table_list[-1].setEditTriggers(QTableView.NoEditTriggers)
		self.table_list[-1].setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

		layout.addWidget(self.table_list[-1])

		tab.setLayout(layout)

	def update_result(self, table_num, result:dict):

		self.table_list[table_num].clear()

		count = 0

		self.table_list[table_num].verticalHeader().setVisible(False)
		self.table_list[table_num].horizontalHeader().setVisible(False)

		self.table_list[table_num].setRowCount(len(result))
		self.table_list[table_num].setColumnCount(2)

		for i in result:
			item1 = QTableWidgetItem(i)
			item2 = QTableWidgetItem( str( result[i]) )
			self.table_list[table_num].setItem(count, 0, item1)
			self.table_list[table_num].setItem(count, 1, item2)
			count += 1

		self.table_list[table_num].resizeColumnsToContents()

	def update_result_tuple(self, table_num:int, result:list, table_size:int):

		count = 0

		header = []

		self.table_list[table_num].clear()

		self.table_list[table_num].horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)#or Stretch

		self.table_list[table_num].setRowCount(table_size)
		self.table_list[table_num].setColumnCount(table_size)

		for i in range(0, table_size):
			header.append(str(i))


		self.table_list[table_num].setVerticalHeaderLabels(header)
		self.table_list[table_num].setHorizontalHeaderLabels(header)

		for (r,c,v) in result:
			item = QTableWidgetItem( str(v) )
			self.table_list[table_num].setItem( int(r), int(c), item )   #先行号，后列号

		self.table_list[table_num].resizeColumnsToContents()

'''
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
'''

if __name__ == "__main__":
	app = QApplication(sys.argv)
	content1 = ['ant','ATM','BCEL','daisy','DNS','SPM','simple','test']
	content2 = ["DQN", "duelingDQN", "doubleDQN", "DQN-CNN","PrioritizedReplayDQN"]
	mainwindow = MainWindow(content1, content2)
	mainwindow.show()
	#app.exec_()           #有这行可以控制界面关闭后执行后面的内容，没这行则同时执行

	#a = input("input a:")
	#print(a)

	sys.exit(app.exec_())    #不加这一句，可以让窗口程序和其它程序一起执行。


