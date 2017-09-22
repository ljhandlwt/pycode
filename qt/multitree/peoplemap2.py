# coding=utf-8

from PyQt4.QtGui import *
from PyQt4.QtCore import *

class mymap(QWidget):
    def __init__(self,jhtree,parent=None):
        '''
        jhtree:peopletree2.tree
        '''

        super(mymap,self).__init__(parent)
        
        self.setWindowTitle(self.tr('title'))
        
        self.maxdeep=0 # depth of the tree
        self.jhtree=jhtree
        # init root
        self.jhtree.mname[''].deep=0
        self.jhtree.mname[''].offset=0
        self.jhtree.mname[''].larr=[]
        self.jhtree.mname[''].rarr=[]
        # make (deep,offset) of all nodes
        self.dfs('')
        
    def dfs(self,name):
        ''' make (deep,offset) of sons of this node '''
        person=self.jhtree.mname[name]
        self.maxdeep=max(self.maxdeep,person.deep)
        for i in person.son:
            self.jhtree.mname[i].deep=person.deep+1
            self.dfs(i)

        person.larr=[]
        person.rarr=[]
        if len(person.son)==0: # leaf
            pass  
        elif len(person.son)==1: # single son
            son=self.jhtree.mname[person.son[0]]
            son.offset=0
            person.larr.append(0)
            person.rarr.append(0)
            for i in range(len(son.larr)):
                person.larr.append(son.larr[i])
                person.rarr.append(son.rarr[i])
        else: # multi son
            larr=self.jhtree.mname[person.son[0]].larr[:]
            rarr=self.jhtree.mname[person.son[0]].rarr[:]
            self.jhtree.mname[person.son[0]].offset=0
            for i in range(len(person.son)-1):
                # decide offset of son
                son=self.jhtree.mname[person.son[i+1]]
                maxdist=self.jhtree.mname[person.son[i]].offset
                for j in range(min(len(rarr),len(son.larr))):
                    maxdist=max(maxdist,rarr[j]-son.larr[j])
                son.offset=self.jhtree.mname[person.son[0]].offset+maxdist+2
                # update most left and right of all depth
                for j in range(min(len(larr),len(son.larr))):
                    larr[j]=min(larr[j],son.larr[j]+son.offset)
                    rarr[j]=max(rarr[j],son.rarr[j]+son.offset)
                # find deeper
                if len(son.larr)>len(larr):
                    for j in range(len(larr),len(son.larr)):
                        larr.append(son.larr[j]+son.offset)
                        rarr.append(son.rarr[j]+son.offset)   
            # align offset of all sons
            sumdist=self.jhtree.mname[person.son[-1]].offset # width of sons
            for i in person.son:
                self.jhtree.mname[i].offset-=sumdist/2.0
            # make larr and rarr of this node    
            person.larr=[self.jhtree.mname[person.son[0]].offset]
            person.rarr=[self.jhtree.mname[person.son[-1]].offset]
            for i in range(len(larr)):
                person.larr.append(larr[i]-sumdist/2.0)
                person.rarr.append(rarr[i]-sumdist/2.0)   
        
    def paintEvent(self,e):
        '''
        draw the tree
        note:called by Qt system
        '''

        # decide the size of the window
        maxX=0
        minX=0
        root=self.jhtree.mname['']
        for i in range(len(root.larr)):
            minX=min(minX,root.larr[i])
            maxX=max(maxX,root.rarr[i])
        maxdist=2*max(maxX,-minX)
        size=QSize((maxdist+1)*30+100,(2*self.maxdeep+1)*30+100) # 30 is size of node, 100 is dist between most left(right) and the left(right) of window
        self.resize(size)
        
        # bfs
        queue=[[self.jhtree.mname[''],size.width()/2.0,5]] # 5+2*30 is dist between root and the top of window
        painter=QPainter()
        painter.begin(self)
        painter.setPen(QPen(Qt.black,Qt.DashLine)) # set style of text and line
        
        while len(queue)!=0:
            person=queue[0][0]
            sx=queue[0][1]
            sy=queue[0][2]
            queue.pop(0)

            if person.name == '': # transparent root
               for i in person.son:
                   son=self.jhtree.mname[i]
                   queue.append([son,sx+son.offset*30,sy+60])
               continue    

            painter.setBrush(QBrush(Qt.blue,Qt.SolidPattern)) # set style of node
            painter.drawRect(sx-15,sy-15,30,30) # draw node

            painter.drawText(sx+15,sy-15,self.tr(person.name))

            if len(person.son)!=0:
                painter.drawLine(sx,sy+15,sx,sy+30)
                if len(person.son)>1:
                    painter.drawLine(sx+self.jhtree.mname[person.son[0]].offset*30,sy+30,sx+self.jhtree.mname[person.son[-1]].offset*30,sy+30) # draw the horizon line
            for i in person.son:
                son=self.jhtree.mname[i]
                painter.drawLine(sx+son.offset*30,sy+30,sx+son.offset*30,sy+45) # draw vertical line
                queue.append([son,sx+son.offset*30,sy+60])
        painter.end()
