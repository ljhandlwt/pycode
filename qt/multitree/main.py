# coding=utf-8

'''
jh:2017.9.1

environment:
py3
pyqt4

peoplemap: GUI
peopletree: data struct

file format:
the first line is the root_name
other lines with node_name and son_name, split with space

note: node_name must appear on the son_name of a line before this line
'''

from PyQt4.QtGui import *
from PyQt4.QtCore import *
import os,sys
import peoplemap2 as peoplemap
import peopletree2 as peopletree
import codecs
import argparse

#support chinese
code=QTextCodec.codecForName("utf8")
QTextCodec.setCodecForTr(code)
QTextCodec.setCodecForCStrings(code)
QTextCodec.setCodecForLocale(code)


def fileload(filepath):
	jhtree = peopletree.tree()

	with codecs.open(filepath, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip().split()
			if len(line) == 1: #root
				jhtree.insert(line[0], '')
			else:	
				jhtree.insert(line[1], line[0])

	return jhtree


def filesave(filepath, jhtree):
	with open(filepath, 'w', encoding='utf-8') as f:
		queue=[jhtree.root]

		while len(queue)!=0:
			person=queue[0]
			queue.pop(0)

			if person.name!='':
				qfile.write(person.name+' '+person.father+'\n')
			for i in person.son:
				queue.append(jhtree.mname[i])


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--file', default='test.txt', help='input tree map')
	args = parser.parse_args()
	filepath = args.file

	jhtree = fileload(filepath)

	app=QApplication(sys.argv)

	gwin=peoplemap.mymap(jhtree)
	gwin.show()

	app.exec_()


if __name__ == '__main__':
	main()	