# coding=utf-8

# node of tree
class peo(object):
    def __init__(self, name, father):
        self.name=name
        self.father=father # name of father
        self.son=[] # name of sons
        # there are some attr will be defined by peoplemap 

        # using (deep,offset) can decide (x,y)
        # self.deep : int, the depth of this node
        # self.offset : int, offset relative to father

        # using in calc (deep,offset)
        # self.larr : list, offset of most left son every depth relative to self
        # self.rarr : list, offset of most right son every depth relative to self
        
class tree(object):
    def __init__(self):
        self.root=peo('','') # define transparent root
        self.mname={'':self.root} # name->node
    
    def insert(self,name,father):
        # make sure no different nodes have same name
        if name in self.mname:
            raise Exception('name existed!')
        self.mname[name]=peo(name,father)
        self.mname[father].son.append(name)
    
    def delete(self,name):
        ''' it will delete all child nodes of this node '''
        person=self.mname[name]
        father=self.mname[person.father]
        father.son.remove(name)
        # bfs
        queue=[name]
        while len(queue)!=0:
            name=queue[0]
            queue.pop(0)
            person=self.mname[name]
            for i in person.son:
                queue.append(i)
            self.mname.pop(name)
    
    def find(self,name):
        ''' name->node '''
        # make sure name is existed
        if not self.is_find(name):
            raise Exception('name not existed')
        return self.mname[name]
    
    def is_find(self,name):
        ''' name->bool'''
        return name in self.mname  